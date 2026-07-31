"""Typed data exchanged across application boundaries."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str | None = None
    published: date | None = None
    doi: str | None = None

    @computed_field
    @property
    def arxiv_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class CitationEvidence(BaseModel):
    source_arxiv_id: str
    target_arxiv_id: str
    contexts: list[str] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    is_influential: bool


class CitationPath(BaseModel):
    citations: list[CitationEvidence] = Field(min_length=1)


class CandidateDiscovery(BaseModel):
    depth: Literal[1, 2]
    paths: list[CitationPath] = Field(min_length=1)


class PrerequisiteCandidate(BaseModel):
    paper: Paper
    discovery: CandidateDiscovery
    supporting_papers: list[Paper] = Field(default_factory=list)


class PrerequisiteRelation(StrEnum):
    ESSENTIAL = "essential"
    HELPFUL = "helpful"
    RELATED_ONLY = "related_only"
    NOT_RELEVANT = "not_relevant"


class PrerequisiteJudgment(BaseModel):
    candidate_id: str
    relation: PrerequisiteRelation
    evidence: str = Field(min_length=1)


class CandidateFailure(BaseModel):
    candidate_id: str
    retrieval_depth: Literal[1, 2]
    attempts: int = Field(ge=1)
    error: str = Field(min_length=1)


class ClassificationResult(BaseModel):
    judgments: list[PrerequisiteJudgment]
    failures: list[CandidateFailure]


class RelationshipProvenance(BaseModel):
    classifier: Literal["model"] = "model"
    candidate_source: Literal["semantic_scholar_citation_graph"] = "semantic_scholar_citation_graph"
    retrieval_depth: Literal[1, 2]
    paths: list[CitationPath] = Field(min_length=1)


class PrerequisiteRelationship(BaseModel):
    source_arxiv_id: str
    target_arxiv_id: str
    relation: PrerequisiteRelation
    evidence: str
    provenance: RelationshipProvenance


class GenerationMetadata(BaseModel):
    model: str
    generated_at: datetime
    complete: bool = True
    failed_candidates: list[CandidateFailure] = Field(default_factory=list)


class KnowledgeMap(BaseModel):
    target_arxiv_id: str
    papers: list[Paper]
    relationships: list[PrerequisiteRelationship]
    generation: GenerationMetadata


class KnowledgeMapRequest(BaseModel):
    arxiv_id_or_url: str = Field(min_length=1)
