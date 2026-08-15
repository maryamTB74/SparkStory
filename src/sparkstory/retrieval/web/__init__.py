"""Grounding a claim the corpus cannot support.

A premise about submarines finds nothing in ``corpus/``, and the model's own
knowledge is deliberately destroyed by ``drop_unprovenanced`` -- an invented
``chunk_id`` does not resolve, so the fact goes. That is the right rule: a claim
we cannot point at is one we will not stand behind in a book written for a named
five-year-old. This package does not weaken it. It gives such claims a source.

**Two steps, and the second is the one that matters.**

1. ``providers`` searches the web and returns candidate claims, each with a URL.
   That URL is written by the *model* into a structured field, so at this point
   it is an assertion. A plausible fabrication is indistinguishable from a real
   citation, which is exactly the defect that made this project overwrite a
   fact's ``source`` from the store rather than trust what the agent wrote.
2. ``verify`` fetches the page and checks that it says what was claimed. A
   fabricated URL cannot survive being fetched, and a real URL that does not
   support the claim is dropped too.

Only what survives both enters the ``ledger``, which is the web counterpart of
``LocalVectorStore``: the thing a ``web:<n>`` id resolves against, and the
authority on attribution. ``drop_unprovenanced`` consults store and ledger alike,
so a web fact is kept or dropped by the same rule as a corpus one.

**The whole package is inert at ``MAX_WEB_SEARCHES=0``**, which is the default.
Nothing here is imported, no client is constructed and neither key is read, so
the test suite keeps its no-network property.
"""
