"""Parameters owned by the public graph projection."""

from knowledge_maps.schemas import PrerequisiteRelation

# The public graph contains learning relationships, while related and rejected
# candidates remain classification outcomes rather than graph edges.
INCLUDED_RELATIONS = {
    PrerequisiteRelation.ESSENTIAL,
    PrerequisiteRelation.HELPFUL,
}
