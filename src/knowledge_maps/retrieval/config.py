"""Parameters owned by citation-graph candidate retrieval."""

from knowledge_maps.schemas import PrerequisiteRelation

# Only direct papers judged to prepare the reader are useful bridges to the
# second citation hop. Related and irrelevant papers are not expanded.
EXPANSION_RELATIONS = {
    PrerequisiteRelation.ESSENTIAL,
    PrerequisiteRelation.HELPFUL,
}
