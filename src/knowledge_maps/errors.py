class KnowledgeMapsError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(KnowledgeMapsError):
    """Raised when required application configuration is missing."""


class ExternalServiceError(KnowledgeMapsError):
    """Raised when an external service returns an unusable response."""


class PaperNotFoundError(KnowledgeMapsError):
    """Raised when a requested paper cannot be resolved."""


class ModelOutputError(KnowledgeMapsError):
    """Raised when model output violates the prerequisite contract."""


class CheckpointError(KnowledgeMapsError):
    """Raised when saved model progress cannot be read or written."""


class TransientExternalServiceError(ExternalServiceError):
    """Raised for an external failure that is safe to retry."""
