"""Parameters owned by citation-graph candidate retrieval."""

from knowledge_maps.schemas import PrerequisiteRelation

# Helpful papers remain optional leaves. Only essential papers are expanded,
# which keeps deeper retrieval focused without imposing a candidate-count cap.
EXPANSION_RELATIONS = {
    PrerequisiteRelation.ESSENTIAL,
}
