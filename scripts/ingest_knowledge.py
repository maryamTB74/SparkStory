"""Build the knowledge index from ``corpus/``.

Offline, idempotent, and the only thing that ever writes ``data/knowledge/``. Run
it once before a grounded story run, and again whenever the corpus changes::

    uv run python scripts/ingest_knowledge.py
    uv run python scripts/ingest_knowledge.py --dry-run   # parse only, no embedding
    uv run python scripts/ingest_knowledge.py --check "could a flag wave on the moon?"

The corpus files are the source of truth; the index is a build artifact and can be
deleted at any time. ``data/`` is gitignored -- unlike ``outputs/``, which holds
disposable debugging output, it holds real persistence, so nothing throwaway goes
here.

A ``--check`` query at the end is worth using: an index that built without error but
retrieves the wrong chunk is the failure that otherwise shows up much later, inside
a story, as a fact that does not fit.
"""

import argparse
import logging
from pathlib import Path

from sparkstory.config import settings
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.embed import get_embedder
from sparkstory.retrieval.hybrid import HybridIndex
from sparkstory.retrieval.ingest import load_corpus
from sparkstory.retrieval.store import LocalVectorStore
from sparkstory.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)

#: Anchored to the repository, not the working directory -- same reasoning as
#: `knowledge_root` in config.py, and the same failure if it were relative.
_DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Where to write the index (default: {settings.knowledge_root}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report, but do not embed or write anything.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="After building, search for this and print the top hits. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    chunks = load_corpus(args.corpus)
    by_kind = {
        kind: [chunk for chunk in chunks if chunk.source_kind is kind]
        for kind in SourceKind
    }

    print(f"corpus: {args.corpus}")
    for kind, group in by_kind.items():
        files = sorted({chunk.chunk_id.split("#")[0] for chunk in group})
        print(f"  {kind.value:6} {len(group):3} chunks from {len(files)} files")
    licences = sorted({chunk.licence for chunk in chunks})
    print(f"  licences: {', '.join(licences)}")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    root = args.out or settings.knowledge_root
    embedder = get_embedder(settings.embedding_model)
    store = LocalVectorStore(root=root, embedder=embedder)
    store.save(chunks)
    print(f"\nwrote {len(chunks)} chunks to {root}")
    print(f"  embedder: {settings.embedding_model} ({embedder.dimensions} dimensions)")

    # Two default probes, one per index, so a build always demonstrates that the
    # thing it built answers. Cheap, and it turns "it ran" into "it works".
    queries = args.check or [
        "could a flag wave on the moon?",
        "a technique that repeats a line so a child can join in",
    ]
    index = HybridIndex(store=store)
    for query in queries:
        print(f"\n  {query!r}")
        for hit in index.search(query, top_k=3):
            preview = hit.chunk.text.replace("\n", " ")[:78]
            print(f"    [{hit.chunk.chunk_id:22}] {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
