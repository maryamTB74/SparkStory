"""Configure the Opik client, or explain why it is off."""

import logging

from sparkstory.config import settings

logger = logging.getLogger(__name__)


def configure() -> bool:
    """Configure Opik if it is enabled and has what it needs.

    Returns:
        True if Opik was configured and tracing can proceed, False otherwise.
        Never raises: every failure is a warning, because the alternative is
        losing a book to an observability problem.
    """
    if not settings.opik_enabled:
        return False

    if settings.opik_api_key is None:
        logger.warning(
            "OPIK_ENABLED is true but OPIK_API_KEY is not set. Tracing is off."
        )
        return False

    # Imported here, not at module scope: see the package docstring.
    import opik

    try:
        opik.configure(
            api_key=settings.opik_api_key.get_secret_value(),
            workspace=settings.opik_workspace,
            use_local=False,
            force=True,
            automatic_approvals=True,
        )
    # Broad by intention. Anything opik raises -- auth, network, or a difference
    # in its own config handling between versions -- must degrade to "no
    # tracing" rather than reaching a caller who asked for a storybook.
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not configure Opik, tracing is off: %s", error)
        return False

    logger.info("Opik configured, project %r", settings.opik_project_name)
    return True
