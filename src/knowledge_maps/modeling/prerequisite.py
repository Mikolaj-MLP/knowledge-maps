import hashlib
import json
import logging
import time
from typing import Protocol

import httpx
from pydantic import ValidationError

from knowledge_maps.errors import (
    CheckpointError,
    ExternalServiceError,
    ModelOutputError,
    TransientExternalServiceError,
)
from knowledge_maps.modeling.config import (
    HUGGING_FACE_COOLDOWN_SECONDS,
    HUGGING_FACE_LIMIT_MESSAGE,
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
    TRANSIENT_HTTP_STATUS_CODES,
    TRANSIENT_RETRY_DELAYS_SECONDS,
)
from knowledge_maps.schemas import (
    CandidateFailure,
    CitationEvidence,
    ClassificationResult,
    Paper,
    PrerequisiteCandidate,
    PrerequisiteJudgment,
)
from knowledge_maps.storage.checkpoints import JudgmentCheckpointStore

LOGGER = logging.getLogger(__name__)


class _HuggingFaceCooldownRequired(TransientExternalServiceError):
    pass


class PrerequisiteModel(Protocol):
    @property
    def name(self) -> str: ...

    def classify(
        self,
        target: Paper,
        candidates: list[PrerequisiteCandidate],
    ) -> ClassificationResult: ...


class OpenAICompatiblePrerequisiteModel:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        http_client: httpx.Client,
        checkpoint_store: JudgmentCheckpointStore,
    ) -> None:
        self._completion_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model_name = model_name
        self._authorization = f"Bearer {api_key}"
        self._http_client = http_client
        self._checkpoint_store = checkpoint_store

    @property
    def name(self) -> str:
        return self._model_name

    def classify(
        self,
        target: Paper,
        candidates: list[PrerequisiteCandidate],
    ) -> ClassificationResult:
        judgments = []
        failures = []

        for candidate in candidates:
            fingerprint = _classification_fingerprint(self.name, target, candidate)
            cached = self._checkpoint_store.get(fingerprint)
            if cached is not None:
                if cached.candidate_id != candidate.paper.arxiv_id:
                    raise CheckpointError("Saved judgment belongs to the wrong candidate")
                judgments.append(cached)
                continue

            attempts = 0
            while True:
                attempts += 1
                try:
                    judgment = self._classify_candidate(target, candidate)
                except _HuggingFaceCooldownRequired as error:
                    if attempts == 1:
                        LOGGER.warning(
                            "Hugging Face request limit reached; retrying %s in %s seconds",
                            candidate.paper.arxiv_id,
                            HUGGING_FACE_COOLDOWN_SECONDS,
                        )
                        time.sleep(HUGGING_FACE_COOLDOWN_SECONDS)
                        continue
                    failures.append(_candidate_failure(candidate, attempts, error))
                    break
                except TransientExternalServiceError as error:
                    if attempts <= len(TRANSIENT_RETRY_DELAYS_SECONDS):
                        delay = TRANSIENT_RETRY_DELAYS_SECONDS[attempts - 1]
                        LOGGER.warning(
                            "Transient model failure for %s; retrying in %s second(s)",
                            candidate.paper.arxiv_id,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    failures.append(_candidate_failure(candidate, attempts, error))
                    break
                except (ExternalServiceError, ModelOutputError) as error:
                    failures.append(_candidate_failure(candidate, attempts, error))
                    break

                self._checkpoint_store.save(
                    fingerprint,
                    self.name,
                    target.arxiv_id,
                    judgment,
                )
                judgments.append(judgment)
                break

        return ClassificationResult(judgments=judgments, failures=failures)

    def _classify_candidate(
        self,
        target: Paper,
        candidate: PrerequisiteCandidate,
    ) -> PrerequisiteJudgment:
        try:
            response = self._http_client.post(
                self._completion_url,
                headers={"Authorization": self._authorization},
                json={
                    "model": self._model_name,
                    "max_tokens": MAX_COMPLETION_TOKENS,
                    "temperature": TEMPERATURE,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "PrerequisiteJudgment",
                            "schema": PrerequisiteJudgment.model_json_schema(),
                            "strict": True,
                        },
                    },
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(_model_input(target, candidate)),
                        },
                    ],
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
            raise TransientExternalServiceError("Model endpoint connection failed") from error

        if _requires_hugging_face_cooldown(response):
            raise _HuggingFaceCooldownRequired(_model_http_error(response))
        if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
            raise TransientExternalServiceError(_model_http_error(response))
        if response.status_code != httpx.codes.OK:
            raise ExternalServiceError(_model_http_error(response))

        content = _completion_content(response)
        try:
            judgment = PrerequisiteJudgment.model_validate_json(content)
        except ValidationError as error:
            raise ModelOutputError("Model output does not match the judgment schema") from error

        if judgment.candidate_id != candidate.paper.arxiv_id:
            raise ModelOutputError("Model returned a judgment for the wrong candidate")
        return judgment


def _candidate_failure(
    candidate: PrerequisiteCandidate,
    attempts: int,
    error: Exception,
) -> CandidateFailure:
    return CandidateFailure(
        candidate_id=candidate.paper.arxiv_id,
        retrieval_depth=candidate.discovery.depth,
        attempts=attempts,
        error=str(error),
    )


def _classification_fingerprint(
    model: str,
    target: Paper,
    candidate: PrerequisiteCandidate,
) -> str:
    payload = {
        "model": model,
        "system_prompt": _SYSTEM_PROMPT,
        "input": _model_input(target, candidate),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _model_input(
    target: Paper,
    candidate: PrerequisiteCandidate,
) -> dict[str, object]:
    citations: dict[tuple[str, str], CitationEvidence] = {}
    citation_paths = []
    for path in candidate.discovery.paths:
        for citation in path.citations:
            citations[(citation.source_arxiv_id, citation.target_arxiv_id)] = citation
        citation_paths.append(
            [
                path.citations[0].source_arxiv_id,
                *(citation.target_arxiv_id for citation in path.citations),
            ]
        )

    return {
        "target": target.model_dump(mode="json"),
        "supporting_papers": [
            paper.model_dump(mode="json")
            for paper in sorted(
                candidate.supporting_papers,
                key=lambda paper: paper.arxiv_id,
            )
        ],
        "citation_evidence": [citation.model_dump(mode="json") for citation in citations.values()],
        "candidate": {
            "paper": candidate.paper.model_dump(mode="json"),
            "retrieval_depth": candidate.discovery.depth,
            "citation_paths": citation_paths,
        },
    }


def _completion_content(response: httpx.Response) -> str:
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ExternalServiceError("Model endpoint returned an invalid completion") from error
    if not isinstance(content, str):
        raise ExternalServiceError("Model completion content is not text")
    return content


def _model_http_error(response: httpx.Response) -> str:
    message = _model_error_message(response)

    result = f"Model endpoint returned HTTP {response.status_code}"
    return f"{result}: {message}" if message else result


def _requires_hugging_face_cooldown(response: httpx.Response) -> bool:
    message = _model_error_message(response)
    return (
        response.status_code == 402
        and message is not None
        and message.startswith(HUGGING_FACE_LIMIT_MESSAGE)
    )


def _model_error_message(response: httpx.Response) -> str | None:
    message = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str):
            message = error
        elif isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        elif isinstance(payload.get("message"), str):
            message = payload["message"]
    return message


_SYSTEM_PROMPT = """\
You classify whether an earlier scientific paper is a prerequisite for understanding a target paper.

Return one judgment for the supplied candidate. Use this JSON shape:
{
  "candidate_id": "arXiv ID",
  "relation": "essential|helpful|related_only|not_relevant",
  "evidence": "brief evidence grounded in the supplied paper metadata"
}

Relations:
- essential: understanding a central contribution of the target reasonably requires this paper.
- helpful: the paper materially prepares the reader, but is not required.
- related_only: scientifically connected, cited, compared, or contemporary, but not preparatory.
- not_relevant: it does not materially help the reader understand the target.

Do not equate citation count, fame, or topical similarity with prerequisite status.

The candidate includes:
- paper: metadata for the paper being judged;
- retrieval_depth: one or two backward citation hops from the target;
- citation_paths: paper IDs ordered from the candidate through any intermediate paper
  to the target.

Supporting_papers contains metadata for intermediate papers. Citation_evidence contains
the evidence for every edge used by the supplied paths. A citation's source paper is cited
by its target paper. Contexts are passages written by that target paper about the source
paper. Citation intents and influential-citation flags are retrieval evidence, not
prerequisite labels.

Always judge the candidate against the original target. A two-hop citation path explains
why the candidate was retrieved; it does not prove that the candidate is a prerequisite.
"""
