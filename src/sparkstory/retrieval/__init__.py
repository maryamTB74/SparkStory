"""Local retrieval: a hybrid index over a small curated corpus.

No service and no server. The index is one ``.npy`` of vectors beside one
``.json`` of chunks, searched with numpy. At this corpus size a served vector
database buys nothing that numpy does not already do.

Nothing here calls a network at query time. The embedding model runs locally, so
retrieval works with no API key and the whole layer is testable offline.
"""
