# video-transcribe

Generated outputs:

- `transcript.txt` - readable transcript grouped by source video part and speaker.
- `transcript.json` - structured transcript with timestamps, source part metadata, speaker labels, and word timings.

Re-run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/transcribe_conversation.py
```
