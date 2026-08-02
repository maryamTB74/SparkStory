# corpus/

The knowledge the Researcher can retrieve. Committed as text, deliberately.

Two directories, and the directory name *is* the `source_kind`:

```
corpus/facts/    things a story must not get wrong   -> source_kind = fact
corpus/craft/    read-aloud techniques               -> source_kind = craft
```

Build the index from these files with:

```bash
uv run python scripts/ingest_knowledge.py
```

That writes `data/knowledge/`, which is gitignored. These files are the source of
truth; the index is a build artifact and can be deleted at any time.

## File format

YAML-ish front-matter, then chunks separated by a line containing only `---`.

```markdown
---
title: The Moon
source: NASA -- Moon facts
licence: public domain
---

One chunk. One fact, in a sentence a five-year-old could hear.

---

The next chunk.
```

`title`, `source` and `licence` are required. `url` is optional and **must be left
out unless the address is known to be correct** — a plausible-looking fabricated
citation is worse than no citation in a feature whose entire purpose is factual
accuracy.

Chunk ids are positional: the third chunk of `moon.md` is `moon#3`, permanently.
So **append rather than insert** — inserting a chunk renumbers every chunk after
it, and a `GroundedFact` recorded in an earlier run cites the old id.

## Why it is small

Around 50 chunks, hand-written. That is a design decision, not a starting point to
grow out of quickly. A large scraped corpus makes every retrieval failure
unattributable, which is the mistake this project has paid for since Session 2 —
and a corpus you can read in full means a bad result has a findable cause. Grow it
*after* `make test-corpus` can measure whether growth helped.

## Licences

- **NASA** material is public domain (US government work).
- **Simple English Wikipedia** is CC BY-SA 4.0, so the `source` field carries the
  attribution and is passed through to `GroundedFact.source`.
- **Project Gutenberg** nursery rhymes are public domain by age.

Facts here are written in our own words from these sources rather than copied, so
what is being attributed is the *fact*, not the phrasing.

## Not Aesop

Fables were considered for `craft/` and rejected: every one ends in a stated moral,
and the planner's hardest rule is *never moralise*. Retrieving moralising exemplars
into a prompt that forbids moralising is a rubric fighting itself.
