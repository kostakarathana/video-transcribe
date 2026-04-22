#!/usr/bin/env python3
"""Transcribe ordered video parts and separate three recurring speakers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import librosa
import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from speechbrain.inference.speaker import EncoderClassifier


AUDIO_SAMPLE_RATE = 16_000
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_PROMPT = (
    "This is a heated family conversation between a daughter, her father, "
    "and her mother. Transcribe the conversation accurately in English."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean speaker-separated transcript from ordered video parts."
    )
    parser.add_argument("--input-dir", default="raw-videos", type=Path)
    parser.add_argument("--work-dir", default="work", type=Path)
    parser.add_argument("--output-txt", default="transcript.txt", type=Path)
    parser.add_argument("--output-json", default="transcript.json", type=Path)
    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-turbo",
        help="MLX Whisper model repository or local path.",
    )
    parser.add_argument(
        "--speaker-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain speaker embedding model.",
    )
    parser.add_argument("--speaker-count", default=3, type=int)
    parser.add_argument(
        "--overcluster-count",
        default=5,
        type=int,
        help="Temporary acoustic cluster count before collapsing artifacts.",
    )
    parser.add_argument(
        "--min-real-speaker-duration",
        default=20.0,
        type=float,
        help="Minimum seconds for a cluster to be treated as a real speaker.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--force", action="store_true", help="Regenerate cached work files.")
    return parser.parse_args()


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    number = int(match.group()) if match else 10**9
    return number, path.name.lower()


def discover_videos(input_dir: Path) -> list[Path]:
    videos = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS],
        key=natural_key,
    )
    if not videos:
        raise SystemExit(f"No video files found in {input_dir}")
    return videos


def ffprobe_duration(path: Path) -> float:
    value = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(value)


def extract_audio(video_path: Path, audio_path: Path, force: bool) -> None:
    if audio_path.exists() and not force:
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain_json(item) for item in value]
    if isinstance(value, tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def transcribe_audio(audio_path: Path, cache_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if cache_path.exists() and not args.force:
        return json.loads(cache_path.read_text())

    print(f"Transcribing {audio_path.name} with {args.whisper_model}", flush=True)
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=args.whisper_model,
        word_timestamps=True,
        language=args.language,
        verbose=False,
        initial_prompt=DEFAULT_PROMPT,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        temperature=0.0,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(plain_json(result), indent=2, ensure_ascii=False))
    return plain_json(result)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def words_from_transcript(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                token = str(word.get("word", "")).strip()
                if not token:
                    continue
                start = word.get("start")
                end = word.get("end")
                if start is None or end is None or float(end) <= float(start):
                    continue
                words.append(
                    {
                        "word": token,
                        "start": float(start),
                        "end": float(end),
                        "probability": word.get("probability"),
                    }
                )
        else:
            text = clean_text(str(segment.get("text", "")))
            start = segment.get("start")
            end = segment.get("end")
            if text and start is not None and end is not None:
                words.append({"word": text, "start": float(start), "end": float(end)})
    return words


def chunk_words(
    words: list[dict[str, Any]],
    *,
    part: int,
    source_file: str,
    part_offset: float,
    max_gap: float = 0.55,
    min_chunk_seconds: float = 1.0,
    max_chunk_seconds: float = 2.6,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = clean_text(" ".join(word["word"] for word in current))
        if text:
            start = float(current[0]["start"])
            end = float(current[-1]["end"])
            chunks.append(
                {
                    "part": part,
                    "source_file": source_file,
                    "start": start,
                    "end": end,
                    "global_start": part_offset + start,
                    "global_end": part_offset + end,
                    "text": text,
                    "words": [
                        {
                            "word": word["word"],
                            "start": float(word["start"]),
                            "end": float(word["end"]),
                            "probability": word.get("probability"),
                        }
                        for word in current
                    ],
                }
            )
        current.clear()

    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            duration = float(current[-1]["end"]) - float(current[0]["start"])
            if gap > max_gap or duration >= max_chunk_seconds:
                flush()
        current.append(word)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        ends_sentence = current[-1]["word"].endswith((".", "?", "!"))
        if duration >= min_chunk_seconds and ends_sentence:
            flush()
    flush()
    return chunks


def read_audio_slice(audio_path: Path, start: float, end: float, padding: float = 0.18) -> np.ndarray:
    file_info = sf.info(str(audio_path))
    sr = file_info.samplerate
    first = max(0, int((start - padding) * sr))
    last = min(file_info.frames, int((end + padding) * sr))
    samples, _ = sf.read(str(audio_path), start=first, stop=last, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if len(samples) < int(0.9 * sr):
        samples = np.pad(samples, (0, int(0.9 * sr) - len(samples)))
    return samples.astype(np.float32)


def embed_chunk(
    classifier: EncoderClassifier, audio_path: Path, start: float, end: float
) -> np.ndarray:
    samples = read_audio_slice(audio_path, start, end)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform).squeeze().cpu().numpy()
    return embedding.astype(np.float32)


def estimate_pitch(audio_path: Path, start: float, end: float) -> float | None:
    samples = read_audio_slice(audio_path, start, end, padding=0.0)
    if len(samples) < int(0.5 * AUDIO_SAMPLE_RATE):
        return None
    try:
        f0 = librosa.yin(
            samples,
            fmin=65,
            fmax=420,
            sr=AUDIO_SAMPLE_RATE,
            frame_length=2048,
            hop_length=512,
        )
        rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=512)[0]
        limit = min(len(f0), len(rms))
        if limit == 0:
            return None
        f0 = f0[:limit]
        rms = rms[:limit]
        voiced = np.isfinite(f0) & (f0 > 65) & (f0 < 420) & (rms > np.percentile(rms, 45))
        if voiced.sum() < 3:
            return None
        return float(np.median(f0[voiced]))
    except Exception:
        return None


def compute_embeddings(
    chunks: list[dict[str, Any]],
    audio_by_part: dict[int, Path],
    classifier: EncoderClassifier,
    cache_path: Path,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if cached.shape[0] == len(chunks):
            return cached

    embeddings = []
    for index, chunk in enumerate(chunks, start=1):
        if index % 50 == 0:
            print(f"Embedded {index}/{len(chunks)} transcript chunks", flush=True)
        audio_path = audio_by_part[int(chunk["part"])]
        embeddings.append(embed_chunk(classifier, audio_path, chunk["start"], chunk["end"]))

    matrix = np.vstack(embeddings)
    np.save(cache_path, matrix)
    return matrix


def select_real_clusters(
    labels: np.ndarray,
    chunks: list[dict[str, Any]],
    speaker_count: int,
    min_duration: float,
) -> list[int]:
    durations: dict[int, float] = {}
    for chunk, label in zip(chunks, labels):
        duration = float(chunk["end"]) - float(chunk["start"])
        durations[int(label)] = durations.get(int(label), 0.0) + max(duration, 0.0)

    retained = [
        label
        for label, _ in sorted(durations.items(), key=lambda item: item[1], reverse=True)
        if durations[label] >= min_duration
    ][:speaker_count]
    if len(retained) < speaker_count:
        for label, _ in sorted(durations.items(), key=lambda item: item[1], reverse=True):
            if label not in retained:
                retained.append(label)
            if len(retained) == speaker_count:
                break
    return retained


def reassign_artifact_clusters(
    labels: np.ndarray,
    matrix: np.ndarray,
    retained_labels: list[int],
) -> np.ndarray:
    retained = set(retained_labels)
    centroids: dict[int, np.ndarray] = {}
    for label in retained_labels:
        centroid = matrix[labels == label].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[label] = centroid / norm if norm else centroid

    reassigned = labels.copy()
    for index, label in enumerate(labels):
        if int(label) in retained:
            continue
        similarities = {
            retained_label: float(np.dot(matrix[index], centroid))
            for retained_label, centroid in centroids.items()
        }
        reassigned[index] = max(similarities, key=similarities.get)
    return reassigned


def father_evidence_score(text: str) -> int:
    text = text.lower()
    weighted_patterns = {
        r"\bas your father\b": 40,
        r"\bme as your father\b": 40,
        r"\bnot like your .*mother\b": 12,
        r"\bme and your mother\b": 10,
        r"\byour mother\b": 5,
        r"\byour mum\b": 4,
        r"\byour father\b": 4,
        r"\byour dad\b": 3,
        r"\blook at your dad\b": 8,
        r"\bi moved out of home\b": 8,
        r"\b61st\b": 8,
        r"\bjoined the army\b": 8,
    }
    return sum(weight * len(re.findall(pattern, text)) for pattern, weight in weighted_patterns.items())


def label_roles(
    stats: dict[int, dict[str, Any]],
    labels: np.ndarray,
    chunks: list[dict[str, Any]],
) -> dict[int, str]:
    text_by_label: dict[int, str] = {label: "" for label in stats}
    for chunk, label in zip(chunks, labels):
        text_by_label[int(label)] += " " + chunk["text"]

    father_scores = {
        label: father_evidence_score(text)
        for label, text in text_by_label.items()
    }
    best_father_label = max(
        stats,
        key=lambda label: (father_scores[label], stats[label]["duration"]),
    )
    if father_scores[best_father_label] == 0:
        best_father_label = max(stats, key=lambda label: stats[label]["duration"])

    remaining = [label for label in stats if label != best_father_label]
    remaining.sort(key=lambda label: stats[label]["duration"], reverse=True)

    role_by_label: dict[int, str] = {best_father_label: "Father"}
    if remaining:
        role_by_label[remaining[0]] = "Daughter"
    if len(remaining) > 1:
        role_by_label[remaining[1]] = "Mother"
    for label in stats:
        role_by_label.setdefault(label, f"Speaker {label + 1}")
    return role_by_label


def assign_speakers(
    chunks: list[dict[str, Any]],
    audio_by_part: dict[int, Path],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    print("Loading speaker embedding model", flush=True)
    classifier = EncoderClassifier.from_hparams(
        source=args.speaker_model,
        savedir=str(args.work_dir / "models" / "speechbrain-spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )

    embedding_cache = args.work_dir / "chunk_embeddings.npy"
    embeddings = compute_embeddings(chunks, audio_by_part, classifier, embedding_cache, args.force)
    matrix = normalize(embeddings)
    if len(chunks) < args.speaker_count:
        labels = np.arange(len(chunks))
    else:
        candidate_count = min(
            len(chunks),
            max(args.speaker_count, args.overcluster_count),
        )
        clustering = AgglomerativeClustering(
            n_clusters=candidate_count,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(matrix)
        retained_labels = select_real_clusters(
            labels,
            chunks,
            args.speaker_count,
            args.min_real_speaker_duration,
        )
        labels = reassign_artifact_clusters(labels, matrix, retained_labels)

    stats: dict[int, dict[str, Any]] = {
        label: {"cluster": f"SPEAKER_{label:02d}", "duration": 0.0, "pitches": []}
        for label in sorted(set(int(label) for label in labels))
    }
    for chunk, label in zip(chunks, labels):
        label = int(label)
        audio_path = audio_by_part[int(chunk["part"])]
        duration = float(chunk["end"]) - float(chunk["start"])
        stats[label]["duration"] += max(duration, 0.0)
        pitch = estimate_pitch(audio_path, chunk["start"], chunk["end"])
        if pitch is not None:
            stats[label]["pitches"].append(pitch)

    for label, values in stats.items():
        pitches = values.pop("pitches")
        values["median_pitch_hz"] = float(np.median(pitches)) if pitches else None

    role_by_label = label_roles(stats, labels, chunks)

    speakers: dict[str, dict[str, Any]] = {}
    for label, role in role_by_label.items():
        key = role.lower().replace(" ", "_")
        speakers[key] = {
            "id": key,
            "label": role,
            "cluster": stats[label]["cluster"],
            "duration_seconds": round(float(stats[label]["duration"]), 3),
            "median_pitch_hz": (
                None
                if stats[label]["median_pitch_hz"] is None
                else round(float(stats[label]["median_pitch_hz"]), 2)
            ),
        }

    for chunk, label in zip(chunks, labels):
        label = int(label)
        chunk["speaker"] = role_by_label[label]
        chunk["speaker_id"] = chunk["speaker"].lower().replace(" ", "_")
        chunk["speaker_cluster"] = stats[label]["cluster"]
    return chunks, speakers


def seconds_to_timecode(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        same_turn = (
            previous
            and previous["part"] == chunk["part"]
            and previous["speaker_id"] == chunk["speaker_id"]
            and chunk["start"] - previous["end"] <= 1.1
            and len(previous["text"]) + len(chunk["text"]) <= 650
        )
        if same_turn:
            previous["end"] = chunk["end"]
            previous["global_end"] = chunk["global_end"]
            previous["text"] = clean_text(previous["text"] + " " + chunk["text"])
            previous["words"].extend(chunk["words"])
        else:
            merged.append(dict(chunk))

    for index, segment in enumerate(merged, start=1):
        segment["index"] = index
        segment["start_time"] = seconds_to_timecode(segment["start"])
        segment["end_time"] = seconds_to_timecode(segment["end"])
        segment["global_start_time"] = seconds_to_timecode(segment["global_start"])
        segment["global_end_time"] = seconds_to_timecode(segment["global_end"])
        segment["duration_seconds"] = round(float(segment["end"]) - float(segment["start"]), 3)
    return merged


def write_txt(path: Path, segments: list[dict[str, Any]], speakers: dict[str, Any]) -> None:
    lines = [
        "Clean Transcript",
        "",
        "Speaker labels are inferred from three voice clusters:",
    ]
    for speaker in sorted(speakers.values(), key=lambda item: item["label"]):
        lines.append(
            f"- {speaker['label']}: {speaker['cluster']}, "
            f"{speaker['duration_seconds']:.1f}s spoken"
        )
    lines.append("")

    current_part: int | None = None
    for segment in segments:
        part = int(segment["part"])
        if part != current_part:
            if current_part is not None:
                lines.append("")
            lines.append(f"Part {part} - {segment['source_file']}")
            current_part = part
        lines.append(
            f"[{segment['start_time']} - {segment['end_time']}] "
            f"{segment['speaker']}: {segment['text']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_json(
    path: Path,
    videos: list[Path],
    part_metadata: list[dict[str, Any]],
    speakers: dict[str, Any],
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "metadata": {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_dir": str(args.input_dir),
            "whisper_model": args.whisper_model,
            "speaker_embedding_model": args.speaker_model,
            "speaker_count": args.speaker_count,
            "overcluster_count": args.overcluster_count,
            "speaker_assignment_note": (
                "Audio was first overclustered to isolate tiny artifact clusters. "
                "Artifacts were reassigned to the nearest real speaker. Father was "
                "labeled from transcript evidence such as 'as your father'; the "
                "remaining longer/shorter real clusters were labeled daughter/mother."
            ),
            "source_files": [str(video) for video in videos],
            "parts": part_metadata,
        },
        "speakers": speakers,
        "segments": segments,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.input_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = args.work_dir / "audio"
    raw_transcript_dir = args.work_dir / "raw-transcripts"
    all_chunks: list[dict[str, Any]] = []
    audio_by_part: dict[int, Path] = {}
    part_metadata: list[dict[str, Any]] = []
    cumulative_offset = 0.0

    for part, video in enumerate(videos, start=1):
        audio_path = audio_dir / f"part_{part:02d}.wav"
        extract_audio(video, audio_path, args.force)
        duration = ffprobe_duration(audio_path)
        audio_by_part[part] = audio_path

        transcript = transcribe_audio(
            audio_path,
            raw_transcript_dir / f"part_{part:02d}.json",
            args,
        )
        words = words_from_transcript(transcript)
        chunks = chunk_words(
            words,
            part=part,
            source_file=str(video),
            part_offset=cumulative_offset,
        )
        all_chunks.extend(chunks)
        part_metadata.append(
            {
                "part": part,
                "source_file": str(video),
                "audio_file": str(audio_path),
                "duration_seconds": round(duration, 3),
                "global_start": round(cumulative_offset, 3),
                "global_end": round(cumulative_offset + duration, 3),
            }
        )
        cumulative_offset += duration

    if not all_chunks:
        raise SystemExit("No transcript chunks were produced.")

    diarized_chunks, speakers = assign_speakers(all_chunks, audio_by_part, args)
    segments = merge_chunks(diarized_chunks)
    write_txt(args.output_txt, segments, speakers)
    write_json(args.output_json, videos, part_metadata, speakers, segments, args)

    print(f"Wrote {args.output_txt}", flush=True)
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
