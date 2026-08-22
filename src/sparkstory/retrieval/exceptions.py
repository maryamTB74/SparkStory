"""Errors raised while building an embedder.

Beside its raiser, exactly as ``models/exceptions.py`` sits beside
``models/get_model.py``. The domain layer stays free of retrieval concerns.

Inherits ``ConfigurationError`` because an operator fixes it by editing
``EMBEDDING_MODEL`` in ``.env`` -- which is what lets the MCP tool layer translate
it into an actionable message while genuine bugs still propagate.
"""

from sparkstory.entities.exceptions import ConfigurationError


class UnknownEmbeddingModelError(ConfigurationError):
    """An embedding model was requested that is not in the registry."""


class EmbeddingDimensionError(ConfigurationError):
    """A provider returned vectors of a width the registry does not declare.

    A ``ConfigurationError`` rather than a bug, because the fix is an operator's:
    the registry says one width and the endpoint produced another, which happens
    when a provider changes a model's default or ignores the width we asked for.

    It exists because the alternative is worse than an error. ``dimensions`` is
    pinned into the pgvector column *and* the table name, so a mismatched vector
    either fails at INSERT -- blaming the database for a model change -- or, if
    the widths happen to agree with some other table, is stored and silently
    ranks against the wrong index.
    """
