import httpx
import pytest

from knowledge_maps.sources.arxiv import ArxivClient, normalize_arxiv_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1706.03762", "1706.03762"),
        ("arXiv:1706.03762v7", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7.pdf", "1706.03762"),
        ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_normalize_arxiv_id(value: str, expected: str) -> None:
    assert normalize_arxiv_id(value) == expected


def test_normalize_arxiv_id_rejects_non_arxiv_url() -> None:
    with pytest.raises(ValueError, match="Invalid arXiv ID or URL"):
        normalize_arxiv_id("https://example.com/paper")


def test_arxiv_sends_id_lists_as_form_data_instead_of_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.query == b""
        assert request.content == b"id_list=2000.00001&max_results=1"
        return httpx.Response(
            200,
            text="""\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2000.00001v1</id>
    <title>Example Paper</title>
    <summary>Example abstract.</summary>
    <published>2020-01-01T00:00:00Z</published>
    <author><name>Example Author</name></author>
  </entry>
</feed>
""",
        )

    client = ArxivClient(httpx.Client(transport=httpx.MockTransport(handler)))

    paper = client.get_paper("2000.00001")

    assert paper.title == "Example Paper"
