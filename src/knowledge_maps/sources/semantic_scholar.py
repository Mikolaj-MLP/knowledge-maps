from urllib.parse import quote

import httpx

from knowledge_maps.errors import ExternalServiceError, PaperNotFoundError
from knowledge_maps.schemas import CitationEvidence, Paper
from knowledge_maps.sources.arxiv import normalize_arxiv_id
from knowledge_maps.sources.config import (
    SEMANTIC_SCHOLAR_PAPER_URL,
    SEMANTIC_SCHOLAR_REFERENCE_FIELDS,
    SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE,
)


class SemanticScholarClient:
    def __init__(self, api_key: str | None, http_client: httpx.Client) -> None:
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._http_client = http_client

    def get_references(self, target: Paper) -> list[CitationEvidence]:
        paper_id = quote(f"ARXIV:{target.arxiv_id}", safe=":")
        offset = 0
        references: list[CitationEvidence] = []

        while True:
            response = self._http_client.get(
                f"{SEMANTIC_SCHOLAR_PAPER_URL}/{paper_id}/references",
                headers=self._headers,
                params={
                    "fields": SEMANTIC_SCHOLAR_REFERENCE_FIELDS,
                    "limit": SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE,
                    "offset": offset,
                },
            )
            if response.status_code == httpx.codes.NOT_FOUND:
                raise PaperNotFoundError(
                    f"Semantic Scholar could not find arXiv paper {target.arxiv_id}"
                )
            payload = _read_payload(response)
            references.extend(_references_from_payload(payload, target.arxiv_id))

            next_offset = payload.get("next")
            if next_offset is None:
                break
            if (
                isinstance(next_offset, bool)
                or not isinstance(next_offset, int)
                or next_offset <= offset
            ):
                raise ExternalServiceError("Semantic Scholar returned invalid pagination")
            offset = next_offset

        return _merge_references(references)


def _read_payload(response: httpx.Response) -> dict[str, object]:
    if response.status_code != httpx.codes.OK:
        raise ExternalServiceError(f"Semantic Scholar returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as error:
        raise ExternalServiceError("Semantic Scholar returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ExternalServiceError("Semantic Scholar returned a non-object response")
    return payload


def _references_from_payload(
    payload: dict[str, object],
    target_arxiv_id: str,
) -> list[CitationEvidence]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ExternalServiceError("Semantic Scholar returned invalid references")

    references = []
    for item in data:
        if not isinstance(item, dict):
            raise ExternalServiceError("Semantic Scholar returned an invalid reference")
        cited_paper = item.get("citedPaper")
        if cited_paper is None:
            continue
        if not isinstance(cited_paper, dict):
            raise ExternalServiceError("Semantic Scholar returned an invalid cited paper")

        cited_arxiv_id = _external_arxiv_id(cited_paper)
        if cited_arxiv_id is None or cited_arxiv_id == target_arxiv_id:
            continue

        contexts = _string_list(item, "contexts")
        intents = _string_list(item, "intents")
        is_influential = item.get("isInfluential")
        if not isinstance(is_influential, bool):
            raise ExternalServiceError(
                "Semantic Scholar returned an invalid influential-citation value"
            )
        references.append(
            CitationEvidence(
                source_arxiv_id=cited_arxiv_id,
                target_arxiv_id=target_arxiv_id,
                contexts=contexts,
                intents=intents,
                is_influential=is_influential,
            )
        )
    return references


def _string_list(value: dict[str, object], field: str) -> list[str]:
    items = value.get(field)
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ExternalServiceError(f"Semantic Scholar returned invalid {field}")
    return [" ".join(item.split()) for item in items if item.strip()]


def _merge_references(references: list[CitationEvidence]) -> list[CitationEvidence]:
    grouped: dict[str, list[CitationEvidence]] = {}
    for reference in references:
        grouped.setdefault(reference.source_arxiv_id, []).append(reference)

    return [
        CitationEvidence(
            source_arxiv_id=arxiv_id,
            target_arxiv_id=items[0].target_arxiv_id,
            contexts=sorted({context for item in items for context in item.contexts}),
            intents=sorted({intent for item in items for intent in item.intents}),
            is_influential=any(item.is_influential for item in items),
        )
        for arxiv_id, items in sorted(grouped.items())
    ]


def _external_arxiv_id(paper: dict[str, object]) -> str | None:
    external_ids = paper.get("externalIds")
    if external_ids is None:
        return None
    if not isinstance(external_ids, dict):
        raise ExternalServiceError("Semantic Scholar returned invalid external IDs")

    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id is None:
        return None
    if not isinstance(arxiv_id, str):
        raise ExternalServiceError("Semantic Scholar returned an invalid arXiv ID")
    try:
        return normalize_arxiv_id(arxiv_id)
    except ValueError as error:
        raise ExternalServiceError("Semantic Scholar returned an invalid arXiv ID") from error
