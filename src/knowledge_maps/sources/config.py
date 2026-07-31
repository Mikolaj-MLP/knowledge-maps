"""Parameters owned by external paper sources."""

ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"

# Ask only for evidence used by candidate discovery and model classification.
SEMANTIC_SCHOLAR_REFERENCE_FIELDS = "contexts,intents,isInfluential,citedPaper.externalIds"

# This is Semantic Scholar's maximum page size, not a candidate limit. The
# client follows pagination until every reference has been read.
SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE = 1000
