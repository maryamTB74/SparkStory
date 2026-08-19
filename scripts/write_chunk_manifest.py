"""Regenerate ``corpus/chunk_manifest.json``.

The manifest records what text each chunk id pointed at when it was last blessed,
so that an *insertion* -- which renumbers every chunk after it -- fails a test
naming exactly what happened, instead of surfacing later as labelled queries
missing their targets and being debugged as a retrieval regression.

Run it only when a change to what an existing id points at is deliberate.
Appending new chunks does not require it, though running it is harmless.
"""

import hashlib
import json
from pathlib import Path

from sparkstory.retrieval.ingest import load_corpus

_CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def main() -> int:
    chunks = load_corpus(_CORPUS)
    manifest = {
        chunk.chunk_id: hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
        for chunk in chunks
    }
    path = _CORPUS / "chunk_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} chunk digests to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
