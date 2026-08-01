import os
from dataclasses import dataclass

from dotenv import load_dotenv

from knowledge_maps.errors import ConfigurationError
from knowledge_maps.modeling.config import DEFAULT_MODEL_NAME, HUGGING_FACE_BASE_URL

# One client is shared by the current sequential pipeline. This timeout applies
# independently to each external request, not to the complete graph run.
HTTP_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Settings:
    semantic_scholar_api_key: str | None
    model_api_key: str
    model_base_url: str
    model_name: str

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(
            semantic_scholar_api_key=_optional_environment_value("S2_API_KEY"),
            model_api_key=_required_environment_value("HF_TOKEN"),
            model_base_url=(_optional_environment_value("MODEL_BASE_URL") or HUGGING_FACE_BASE_URL),
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip(),
        )


def _required_environment_value(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ConfigurationError(f"Required environment variable is missing: {name}")
    return value.strip()


def _optional_environment_value(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None
