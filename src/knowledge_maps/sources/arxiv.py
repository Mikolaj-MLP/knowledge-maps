import re
from datetime import date
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from knowledge_maps.errors import ExternalServiceError, PaperNotFoundError
from knowledge_maps.schemas import Paper
from knowledge_maps.sources.config import ARXIV_API_URL

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
MODERN_ID = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
LEGACY_ID = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?$", re.IGNORECASE)
VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_arxiv_id(value: str) -> str:
    candidate = value.strip()
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:]

    parsed = urlparse(candidate)
    if parsed.netloc.lower() in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        path = parsed.path.strip("/")
        if path.startswith("abs/"):
            candidate = path[4:]
        elif path.startswith("pdf/"):
            candidate = path[4:]
            if candidate.endswith(".pdf"):
                candidate = candidate[:-4]
        else:
            raise ValueError("arXiv URL must point to an abstract or PDF")

    candidate = VERSION_SUFFIX.sub("", candidate)
    if MODERN_ID.fullmatch(candidate) or LEGACY_ID.fullmatch(candidate):
        return candidate
    raise ValueError(f"Invalid arXiv ID or URL: {value}")


class ArxivClient:
    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client

    def get_paper(self, arxiv_id_or_url: str) -> Paper:
        arxiv_id = normalize_arxiv_id(arxiv_id_or_url)
        papers = self.get_papers([arxiv_id])
        return papers[0]

    def get_papers(self, arxiv_ids_or_urls: list[str]) -> list[Paper]:
        arxiv_ids = list(dict.fromkeys(normalize_arxiv_id(value) for value in arxiv_ids_or_urls))
        if not arxiv_ids:
            return []

        response = self._http_client.post(
            ARXIV_API_URL,
            data={
                "id_list": ",".join(arxiv_ids),
                "max_results": len(arxiv_ids),
            },
        )
        if response.status_code != httpx.codes.OK:
            raise ExternalServiceError(f"arXiv returned HTTP {response.status_code}")

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as error:
            raise ExternalServiceError("arXiv returned invalid XML") from error

        papers_by_id = {
            paper.arxiv_id: paper
            for paper in (
                _paper_from_entry(entry) for entry in root.findall("atom:entry", ATOM_NAMESPACE)
            )
        }
        missing_ids = set(arxiv_ids) - papers_by_id.keys()
        if missing_ids:
            missing = ", ".join(sorted(missing_ids))
            raise PaperNotFoundError(f"arXiv papers not found: {missing}")
        return [papers_by_id[arxiv_id] for arxiv_id in arxiv_ids]


def _paper_from_entry(entry: ElementTree.Element) -> Paper:
    entry_id = normalize_arxiv_id(_required_text(entry, "atom:id"))
    title = _required_text(entry, "atom:title")
    abstract = _required_text(entry, "atom:summary")
    published_text = _required_text(entry, "atom:published")
    authors = [
        _required_text(author, "atom:name")
        for author in entry.findall("atom:author", ATOM_NAMESPACE)
    ]
    doi_element = entry.find("{http://arxiv.org/schemas/atom}doi")

    return Paper(
        arxiv_id=entry_id,
        title=_normalize_whitespace(title),
        authors=authors,
        abstract=_normalize_whitespace(abstract),
        published=date.fromisoformat(published_text[:10]),
        doi=doi_element.text.strip() if doi_element is not None and doi_element.text else None,
    )


def _required_text(element: ElementTree.Element, path: str) -> str:
    child = element.find(path, ATOM_NAMESPACE)
    if child is None or child.text is None or not child.text.strip():
        raise ExternalServiceError(f"arXiv response is missing {path}")
    return child.text.strip()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())
