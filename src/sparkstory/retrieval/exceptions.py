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
