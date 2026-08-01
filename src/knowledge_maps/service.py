from collections import Counter, deque
from datetime import UTC, datetime
from time import perf_counter

from knowledge_maps.modeling.prerequisite import PrerequisiteModel
from knowledge_maps.output.graph import build_knowledge_map
from knowledge_maps.retrieval.citation_graph import (
    direct_discoveries,
    expansion_bridge_ids,
    materialize_candidates,
    merge_discoveries,
    next_hop_discoveries,
)
from knowledge_maps.schemas import (
    ExpansionLevelMetrics,
    GenerationMetadata,
    GenerationMetrics,
    KnowledgeMap,
    PrerequisiteRelation,
)
from knowledge_maps.sources.arxiv import ArxivClient
from knowledge_maps.sources.semantic_scholar import SemanticScholarClient


class KnowledgeMapService:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        reference_client: SemanticScholarClient,
        prerequisite_model: PrerequisiteModel,
    ) -> None:
        self._arxiv_client = arxiv_client
        self._reference_client = reference_client
        self._prerequisite_model = prerequisite_model

    def build(self, arxiv_id_or_url: str) -> KnowledgeMap:
        started_at = perf_counter()
        target = self._arxiv_client.get_paper(arxiv_id_or_url)
        papers_by_id = {target.arxiv_id: target}
        expansion_discoveries = {}
        frontier = deque([target.arxiv_id])
        scheduled_paper_ids = {target.arxiv_id}
        expanded_paper_ids = set()

        candidates = []
        judgments = []
        failures = []
        unique_candidate_ids = set()
        inference_requests = 0
        checkpoint_hits = 0
        papers_expanded_by_depth: Counter[int] = Counter()
        candidates_by_depth: Counter[int] = Counter()
        essential_edges_by_depth: Counter[int] = Counter()
        helpful_edges_by_depth: Counter[int] = Counter()
        failures_by_depth: Counter[int] = Counter()

        while frontier:
            parent_id = frontier.popleft()
            parent = papers_by_id[parent_id]
            parent_discovery = expansion_discoveries.get(parent_id)
            parent_depth = parent_discovery.depth if parent_discovery else 0
            references = self._reference_client.get_references(parent)
            discoveries = (
                next_hop_discoveries(parent_discovery, references)
                if parent_discovery
                else direct_discoveries(references)
            )

            expanded_paper_ids.add(parent_id)
            papers_expanded_by_depth[parent_depth] += 1
            missing_ids = sorted(set(discoveries) - papers_by_id.keys())
            if missing_ids:
                papers = self._arxiv_client.get_papers(missing_ids)
                papers_by_id.update({paper.arxiv_id: paper for paper in papers})

            parent_candidates = materialize_candidates(
                discoveries,
                papers_by_id,
                parent_id,
            )
            result = self._prerequisite_model.classify(
                target,
                parent,
                parent_candidates,
            )
            candidates.extend(parent_candidates)
            judgments.extend(result.judgments)
            failures.extend(result.failures)
            unique_candidate_ids.update(discoveries)
            candidates_by_depth[parent_depth] += len(parent_candidates)
            inference_requests += result.inference_requests
            checkpoint_hits += result.checkpoint_hits
            essential_edges_by_depth[parent_depth] += sum(
                judgment.relation is PrerequisiteRelation.ESSENTIAL for judgment in result.judgments
            )
            helpful_edges_by_depth[parent_depth] += sum(
                judgment.relation is PrerequisiteRelation.HELPFUL for judgment in result.judgments
            )
            failures_by_depth[parent_depth] += len(result.failures)

            for bridge_id in expansion_bridge_ids(result.judgments):
                discovery = discoveries[bridge_id]
                previous_discovery = expansion_discoveries.get(bridge_id)
                expansion_discoveries[bridge_id] = (
                    merge_discoveries(previous_discovery, discovery)
                    if previous_discovery
                    else discovery
                )
                if bridge_id not in scheduled_paper_ids:
                    frontier.append(bridge_id)
                    scheduled_paper_ids.add(bridge_id)

        levels = [
            ExpansionLevelMetrics(
                depth=depth,
                papers_expanded=papers_expanded_by_depth[depth],
                candidates_classified=candidates_by_depth[depth],
                essential_edges=essential_edges_by_depth[depth],
                helpful_edges=helpful_edges_by_depth[depth],
                failed_classifications=failures_by_depth[depth],
            )
            for depth in sorted(papers_expanded_by_depth)
        ]
        generation = GenerationMetadata(
            model=self._prerequisite_model.name,
            generated_at=datetime.now(UTC),
            metrics=GenerationMetrics(
                duration_seconds=round(perf_counter() - started_at, 3),
                papers_expanded=len(expanded_paper_ids),
                unique_candidates_considered=len(unique_candidate_ids),
                inference_requests=inference_requests,
                checkpoint_hits=checkpoint_hits,
                levels=levels,
            ),
            complete=not failures,
            failed_candidates=failures,
        )
        return build_knowledge_map(target, candidates, judgments, generation)
