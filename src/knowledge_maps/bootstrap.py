import httpx

from knowledge_maps.config import HTTP_TIMEOUT_SECONDS, Settings
from knowledge_maps.modeling.prerequisite import OpenAICompatiblePrerequisiteModel
from knowledge_maps.service import KnowledgeMapService
from knowledge_maps.sources.arxiv import ArxivClient
from knowledge_maps.sources.semantic_scholar import SemanticScholarClient
from knowledge_maps.storage.checkpoints import JudgmentCheckpointStore
from knowledge_maps.storage.config import CHECKPOINT_DATABASE_PATH


def create_service() -> KnowledgeMapService:
    settings = Settings.from_environment()
    http_client = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    return KnowledgeMapService(
        arxiv_client=ArxivClient(http_client),
        reference_client=SemanticScholarClient(settings.semantic_scholar_api_key, http_client),
        prerequisite_model=OpenAICompatiblePrerequisiteModel(
            settings.model_base_url,
            settings.model_name,
            settings.model_api_key,
            http_client,
            JudgmentCheckpointStore(CHECKPOINT_DATABASE_PATH),
        ),
    )
