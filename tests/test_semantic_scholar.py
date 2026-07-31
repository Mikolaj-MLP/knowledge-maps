import httpx

from knowledge_maps.schemas import Paper
from knowledge_maps.sources.semantic_scholar import SemanticScholarClient


def test_semantic_scholar_returns_paginated_arxiv_references_with_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/paper/ARXIV:2000.00001/references")
        assert (
            request.url.params["fields"] == "contexts,intents,isInfluential,citedPaper.externalIds"
        )
        assert request.url.params["limit"] == "1000"
        assert request.headers["x-api-key"] == "test-key"

        if request.url.params["offset"] == "0":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "contexts": ["First\ncitation context"],
                            "intents": ["background"],
                            "isInfluential": False,
                            "citedPaper": {"externalIds": {"ArXiv": "1900.00002v2"}},
                        },
                        {
                            "contexts": [],
                            "intents": [],
                            "isInfluential": False,
                            "citedPaper": {"externalIds": None},
                        },
                    ],
                    "next": 2,
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "contexts": ["Method context"],
                        "intents": ["methodology"],
                        "isInfluential": True,
                        "citedPaper": {"externalIds": {"ArXiv": "1900.00002"}},
                    },
                    {
                        "contexts": [],
                        "intents": [],
                        "isInfluential": False,
                        "citedPaper": {"externalIds": {"ArXiv": "1900.00001"}},
                    },
                    {
                        "contexts": [],
                        "intents": [],
                        "isInfluential": False,
                        "citedPaper": {"externalIds": {"DOI": "10.1000/journal-only"}},
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = SemanticScholarClient("test-key", client)
    target = Paper(arxiv_id="2000.00001", title="Target Paper", authors=[])

    references = source.get_references(target)

    assert [reference.source_arxiv_id for reference in references] == [
        "1900.00001",
        "1900.00002",
    ]
    assert references[1].target_arxiv_id == target.arxiv_id
    assert references[1].contexts == ["First citation context", "Method context"]
    assert references[1].intents == ["background", "methodology"]
    assert references[1].is_influential is True
