from fastapi import FastAPI, HTTPException

from knowledge_maps.bootstrap import create_service
from knowledge_maps.errors import (
    CheckpointError,
    ExternalServiceError,
    ModelOutputError,
    PaperNotFoundError,
)
from knowledge_maps.schemas import KnowledgeMap, KnowledgeMapRequest
from knowledge_maps.service import KnowledgeMapService


def create_app(service: KnowledgeMapService | None = None) -> FastAPI:
    app = FastAPI(title="Knowledge Maps")
    knowledge_map_service = service or create_service()

    @app.post("/knowledge-maps", response_model=KnowledgeMap)
    def build_knowledge_map(request: KnowledgeMapRequest) -> KnowledgeMap:
        try:
            return knowledge_map_service.build(request.arxiv_id_or_url)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except PaperNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ExternalServiceError, ModelOutputError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except CheckpointError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app
