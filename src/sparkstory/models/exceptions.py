"""Errors raised while building a model.

Separate from ``entities/exceptions.py`` so that the domain layer stays free of
provider concerns: ``entities`` defines what a story is, and knows nothing about
API keys or model registries.

Both inherit ``ConfigurationError`` -- an operator can fix either by editing
``.env`` -- which is what allows the tool layer to translate them into a
client-facing message while letting genuine bugs propagate.
"""

from sparkstory.entities.exceptions import ConfigurationError


class UnknownModelError(ConfigurationError):
    """A model id was requested that is not in the registry."""


class MissingAPIKeyError(ConfigurationError):
    """The API key required by a model is not configured."""
