"""Local retrieval: a hybrid index over a small curated corpus.

No service and no server. The index is one ``.npy`` of vectors beside one
``.json`` of chunks, searched with numpy -- which is what lesson 11 actually does,
whatever the lesson 9 text says about Milvus and Pinecone.

Nothing here calls a network at query time. The embedding model runs locally, so
retrieval works with no API key and the whole layer is testable offline.
"""
