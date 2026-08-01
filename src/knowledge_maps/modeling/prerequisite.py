import hashlib
import json
import logging
import time
from enum import StrEnum
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from knowledge_maps.errors import (
    CheckpointError,
    ExternalServiceError,
    ModelOutputError,
    TransientExternalServiceError,
)
from knowledge_maps.modeling.config import (
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
    PrerequisiteRelation,
)
from knowledge_maps.storage.checkpoints import JudgmentCheckpointStore

LOGGER = logging.getLogger(__name__)


class _CandidateRole(StrEnum):
    CENTRAL_DEPENDENCY = "central_dependency"
    SUPPORTING_BACKGROUND = "supporting_background"
    EXPERIMENTAL_CONTEXT = "experimental_context"
    UNRELATED = "unrelated"


_RELATION_BY_ROLE = {
    _CandidateRole.CENTRAL_DEPENDENCY: PrerequisiteRelation.ESSENTIAL,
    _CandidateRole.SUPPORTING_BACKGROUND: PrerequisiteRelation.HELPFUL,
    _CandidateRole.EXPERIMENTAL_CONTEXT: PrerequisiteRelation.RELATED_ONLY,
    _CandidateRole.UNRELATED: PrerequisiteRelation.NOT_RELEVANT,
}


class _ModelJudgment(BaseModel):
    evidence: str = Field(min_length=1)
    role: _CandidateRole


class PrerequisiteModel(Protocol):
    @property
    def name(self) -> str: ...

    def classify(
        self,
        root_target: Paper,
        immediate_target: Paper,
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
        root_target: Paper,
        immediate_target: Paper,
        candidates: list[PrerequisiteCandidate],
    ) -> ClassificationResult:
        judgments = []
        failures = []
        inference_requests = 0
        checkpoint_hits = 0

        for candidate in candidates:
            fingerprint = _classification_fingerprint(
                self.name,
                root_target,
                immediate_target,
                candidate,
            )
            cached = self._checkpoint_store.get(fingerprint)
            if cached is not None:
                if (
                    cached.candidate_id != candidate.paper.arxiv_id
                    or cached.target_id != immediate_target.arxiv_id
                ):
                    raise CheckpointError("Saved judgment belongs to the wrong relationship")
                checkpoint_hits += 1
                judgments.append(cached)
                continue

            attempts = 0
            while True:
                attempts += 1
                inference_requests += 1
                try:
                    judgment = self._classify_candidate(
                        root_target,
                        immediate_target,
                        candidate,
                    )
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
                    immediate_target.arxiv_id,
                    judgment,
                )
                judgments.append(judgment)
                break

        return ClassificationResult(
            judgments=judgments,
            failures=failures,
            inference_requests=inference_requests,
            checkpoint_hits=checkpoint_hits,
        )

    def _classify_candidate(
        self,
        root_target: Paper,
        immediate_target: Paper,
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
                            "name": "ModelJudgment",
                            "schema": _ModelJudgment.model_json_schema(),
                            "strict": True,
                        },
                    },
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                _model_input(
                                    root_target,
                                    immediate_target,
                                    candidate,
                                )
                            ),
                        },
                    ],
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
            raise TransientExternalServiceError("Model endpoint connection failed") from error

        if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
            raise TransientExternalServiceError(_model_http_error(response))
        if response.status_code != httpx.codes.OK:
            raise ExternalServiceError(_model_http_error(response))

        content = _completion_content(response)
        try:
            model_judgment = _ModelJudgment.model_validate_json(content)
        except ValidationError as error:
            raise ModelOutputError("Model output does not match the judgment schema") from error

        return PrerequisiteJudgment(
            candidate_id=candidate.paper.arxiv_id,
            target_id=immediate_target.arxiv_id,
            relation=_RELATION_BY_ROLE[model_judgment.role],
            evidence=model_judgment.evidence,
        )


def _candidate_failure(
    candidate: PrerequisiteCandidate,
    attempts: int,
    error: Exception,
) -> CandidateFailure:
    return CandidateFailure(
        candidate_id=candidate.paper.arxiv_id,
        target_id=candidate.target_id,
        retrieval_depth=candidate.discovery.depth,
        attempts=attempts,
        error=str(error),
    )


def _classification_fingerprint(
    model: str,
    root_target: Paper,
    immediate_target: Paper,
    candidate: PrerequisiteCandidate,
) -> str:
    payload = {
        "model": model,
        "system_prompt": _SYSTEM_PROMPT,
        "input": _model_input(root_target, immediate_target, candidate),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _model_input(
    root_target: Paper,
    immediate_target: Paper,
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
        "root_target": root_target.model_dump(mode="json"),
        "immediate_target": immediate_target.model_dump(mode="json"),
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
  "evidence": "brief evidence grounded in the supplied paper metadata",
  "role": "central_dependency|supporting_background|experimental_context|unrelated"
}

Roles:
- central_dependency: the target's central method directly extends, replaces, or combines a
  framework, mechanism, result, or problem formulation introduced by this paper.
- supporting_background: the paper teaches a concrete method or theory used by the target, but
  its content is not a central dependency.
- experimental_context: the paper supplies a dataset, metric, optimizer, training setup,
  empirical baseline, comparison, analogy, motivation, or alternative method.
- unrelated: the citation has no material preparatory value for the target's central method.

Root_target is the paper requested by the user. Immediate_target is the paper that
the candidate would prepare the reader to understand. For a direct citation they
are the same paper. For a deeper citation they are different.

Judge whether reading the candidate prepares the reader for the immediate_target,
specifically for the parts of the immediate_target needed to understand root_target.
Reject background that is useful for unrelated parts of the immediate_target.
Do not equate citation count, fame, or topical similarity with prerequisite status.

Build a minimal reading list, not a map of intellectual ancestry. A central dependency must be
part of the target's main technical idea; it is not enough that the target uses, cites, or is
historically descended from it.

Apply these distinctions strictly:
- A paper defining the framework or prior mechanism that the central contribution directly
  extends, replaces, or combines is a central_dependency.
- A broader foundation or supporting theory is supporting_background when it prepares the
  reader but is not itself the object of the target's contribution.
- A dataset, metric, evaluation protocol, empirical baseline, result comparison, analogy,
  motivation, optimizer, training recipe, or alternative method is experimental_context when
  that is its only role. Citation metadata calling it influential or methodological does not
  change that role.
- A citation with no preparatory value is unrelated.

Ask this counterfactual: assuming the reader accepts the target's experimental setup and reads
the explanations contained in the target, would skipping this candidate prevent them from
following the central method? If not, do not classify it as central_dependency. When evidence
is ambiguous, choose the less prerequisite-heavy role.

The candidate includes:
- paper: metadata for the paper being judged;
- retrieval_depth: the number of backward citation hops from the root target;
- citation_paths: paper IDs ordered from the candidate through any intermediate paper
  to the target.

Citation_evidence contains the evidence for every edge used by the supplied paths.
A citation's source paper is cited by its target paper. Contexts are passages written
by that target paper about the source paper. Citation intents and influential-citation
flags are retrieval evidence, not prerequisite labels.

A citation path explains why the candidate was retrieved; it does not prove that the
candidate is a prerequisite.
"""
