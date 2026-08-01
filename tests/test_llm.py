import json
from pathlib import Path

import httpx

from knowledge_maps.modeling.prerequisite import OpenAICompatiblePrerequisiteModel
from knowledge_maps.schemas import (
    CandidateDiscovery,
    CitationEvidence,
    CitationPath,
    Paper,
    PrerequisiteCandidate,
    PrerequisiteRelation,
)
from knowledge_maps.storage.checkpoints import JudgmentCheckpointStore


def test_model_returns_validated_judgments_for_supplied_candidates(tmp_path: Path) -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    candidates = [
        _candidate(
            Paper(arxiv_id="1900.00001", title="Foundation", authors=[]),
            target,
            context="The target uses the foundation's method.",
        ),
        _candidate(
            Paper(arxiv_id="1900.00002", title="Comparison", authors=[]),
            target,
            context="The target compares against this result.",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request_payload["max_tokens"] == 512
        assert request_payload["temperature"] == 0
        assert request_payload["response_format"]["type"] == "json_schema"
        model_input = json.loads(request_payload["messages"][1]["content"])
        candidate_id = model_input["candidate"]["paper"]["arxiv_id"]
        assert model_input["root_target"]["arxiv_id"] == target.arxiv_id
        assert model_input["immediate_target"]["arxiv_id"] == target.arxiv_id
        citation = model_input["citation_evidence"][0]
        assert citation["source_arxiv_id"] == candidate_id
        assert model_input["candidate"]["citation_paths"] == [[candidate_id, "2000.00001"]]
        role = "central_dependency" if candidate_id == "1900.00001" else "experimental_context"
        evidence = (
            "The target directly extends its method."
            if candidate_id == "1900.00001"
            else "It is used only as a comparison."
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "role": role,
                                    "evidence": evidence,
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatiblePrerequisiteModel(
        "http://model.test",
        "qwen",
        "test-token",
        client,
        JudgmentCheckpointStore(tmp_path / "checkpoints.sqlite3"),
    )

    result = model.classify(target, target, candidates)

    assert [judgment.relation for judgment in result.judgments] == [
        PrerequisiteRelation.ESSENTIAL,
        PrerequisiteRelation.RELATED_ONLY,
    ]
    assert result.failures == []
    assert [(judgment.candidate_id, judgment.target_id) for judgment in result.judgments] == [
        (candidate.paper.arxiv_id, target.arxiv_id) for candidate in candidates
    ]
    assert result.inference_requests == 2
    assert result.checkpoint_hits == 0


def test_model_retries_transient_failures_and_reuses_the_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    candidate = _candidate(
        Paper(arxiv_id="1900.00001", title="Foundation", authors=[]),
        target,
    )
    request_count = 0
    delays = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count < 3:
            return httpx.Response(503)
        return _judgment_response()

    monkeypatch.setattr(
        "knowledge_maps.modeling.prerequisite.time.sleep",
        delays.append,
    )
    store = JudgmentCheckpointStore(tmp_path / "checkpoints.sqlite3")
    model = OpenAICompatiblePrerequisiteModel(
        "http://model.test",
        "qwen",
        "test-token",
        httpx.Client(transport=httpx.MockTransport(handler)),
        store,
    )

    first_result = model.classify(target, target, [candidate])
    second_result = model.classify(target, target, [candidate])

    assert request_count == 3
    assert delays == [1, 2]
    assert first_result.failures == []
    assert first_result.inference_requests == 3
    assert first_result.checkpoint_hits == 0
    assert second_result.judgments == first_result.judgments
    assert second_result.inference_requests == 0
    assert second_result.checkpoint_hits == 1


def test_model_records_a_non_transient_failure_and_continues(
    tmp_path: Path,
) -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    failed_candidate = _candidate(
        Paper(arxiv_id="1900.00001", title="Unavailable", authors=[]),
        target,
    )
    successful_candidate = _candidate(
        Paper(arxiv_id="1900.00002", title="Foundation", authors=[]),
        target,
    )
    requested_ids = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        model_input = json.loads(request_payload["messages"][1]["content"])
        candidate_id = model_input["candidate"]["paper"]["arxiv_id"]
        requested_ids.append(candidate_id)
        if candidate_id == failed_candidate.paper.arxiv_id:
            return httpx.Response(402, json={"error": "Provider billing rejected the request"})
        return _judgment_response()

    model = OpenAICompatiblePrerequisiteModel(
        "http://model.test",
        "qwen",
        "test-token",
        httpx.Client(transport=httpx.MockTransport(handler)),
        JudgmentCheckpointStore(tmp_path / "checkpoints.sqlite3"),
    )

    result = model.classify(target, target, [failed_candidate, successful_candidate])

    assert requested_ids == [failed_candidate.paper.arxiv_id, successful_candidate.paper.arxiv_id]
    assert [judgment.candidate_id for judgment in result.judgments] == [
        successful_candidate.paper.arxiv_id
    ]
    assert result.failures[0].candidate_id == failed_candidate.paper.arxiv_id
    assert result.failures[0].attempts == 1
    assert result.failures[0].error == (
        "Model endpoint returned HTTP 402: Provider billing rejected the request"
    )


def test_model_records_invalid_structured_output_as_failed(tmp_path: Path) -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    candidates = [
        _candidate(
            Paper(arxiv_id="1900.00001", title="Foundation", authors=[]),
            target,
        )
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "role": "supporting_background",
                                }
                            ),
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatiblePrerequisiteModel(
        "http://model.test",
        "qwen",
        "test-token",
        client,
        JudgmentCheckpointStore(tmp_path / "checkpoints.sqlite3"),
    )

    result = model.classify(target, target, candidates)

    assert result.judgments == []
    assert result.failures[0].candidate_id == "1900.00001"
    assert result.failures[0].attempts == 1
    assert result.failures[0].error == "Model output does not match the judgment schema"


def test_model_classifies_each_candidate_once_in_an_independent_request(
    tmp_path: Path,
) -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    candidates = [
        _candidate(
            Paper(arxiv_id=f"1900.{index:05d}", title=f"Paper {index}", authors=[]),
            target,
        )
        for index in range(12)
    ]
    request_candidate_ids = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        model_input = json.loads(request_payload["messages"][1]["content"])
        candidate_id = model_input["candidate"]["paper"]["arxiv_id"]
        request_candidate_ids.append([candidate_id])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "role": "supporting_background",
                                    "evidence": "It prepares the reader.",
                                }
                            )
                        }
                    }
                ]
            },
        )

    model = OpenAICompatiblePrerequisiteModel(
        "http://model.test",
        "qwen",
        "test-token",
        httpx.Client(transport=httpx.MockTransport(handler)),
        JudgmentCheckpointStore(tmp_path / "checkpoints.sqlite3"),
    )

    result = model.classify(target, target, candidates)

    assert [len(candidate_ids) for candidate_ids in request_candidate_ids] == [1] * 12
    assert [judgment.candidate_id for judgment in result.judgments] == [
        candidate.paper.arxiv_id for candidate in candidates
    ]
    assert result.failures == []


def _candidate(
    paper: Paper,
    target: Paper,
    context: str = "The target cites this paper.",
) -> PrerequisiteCandidate:
    return PrerequisiteCandidate(
        paper=paper,
        target_id=target.arxiv_id,
        discovery=CandidateDiscovery(
            depth=1,
            paths=[
                CitationPath(
                    citations=[
                        CitationEvidence(
                            source_arxiv_id=paper.arxiv_id,
                            target_arxiv_id=target.arxiv_id,
                            contexts=[context],
                            intents=["background"],
                            is_influential=False,
                        )
                    ]
                )
            ],
        ),
    )


def _judgment_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "role": "supporting_background",
                                "evidence": "It prepares the reader.",
                            }
                        )
                    }
                }
            ]
        },
    )
