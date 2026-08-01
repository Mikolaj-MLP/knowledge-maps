# Knowledge Maps

Knowledge Maps helps readers find the prerequisite arXiv papers they should read
to understand a target paper.

It accepts an arXiv ID or URL and returns a directed prerequisite graph as JSON.

## Pipeline

1. Fetch the target paper's metadata and abstract from arXiv.
2. Retrieve direct references and citation evidence from Semantic Scholar.
3. Classify each candidate as `essential`, `helpful`, `related_only`, or
   `not_relevant`.
4. Expand every `essential` prerequisite and retrieve its references.
5. Classify each deeper candidate against its immediate paper while retaining
   the requested paper and complete citation path as context.
6. Continue until no branch produces another `essential` prerequisite.
7. Return selected papers, prerequisite relationships, model evidence, citation
   paths, and generation metadata.

Each relationship points from a prerequisite to the paper it prepares the
reader for. The resulting graph therefore represents both direct prerequisites
and ordered reading paths.

Traversal has no fixed depth. Each paper is expanded once, citation cycles are
discarded, and `helpful` relationships remain leaves.

## Example

A complete run for
[*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691)
produced this prerequisite graph:

```mermaid
flowchart LR
    transformer["Attention Is All You Need<br/>1706.03762"]
    vocabulary["Strategies for Training Large<br/>Vocabulary Neural Language Models<br/>1512.04906"]
    sublinear["Training Deep Nets with<br/>Sublinear Memory Cost<br/>1604.06174"]
    volta["Dissecting the NVIDIA Volta<br/>GPU Architecture<br/>1804.06826"]
    online["Online Normalizer Calculation<br/>for Softmax<br/>1805.02867"]
    mqa["Fast Transformer Decoding:<br/>One Write-Head Is All You Need<br/>1911.02150"]
    reformer["Reformer:<br/>The Efficient Transformer<br/>2001.04451"]
    butterfly["Learning Fast Algorithms<br/>Using Butterfly Factorizations<br/>1903.05895"]
    keops["Kernel Operations on the GPU,<br/>with Autodiff<br/>2004.11127"]
    movement["Data Movement Is All You Need<br/>2007.00072"]
    memory["Self-Attention Does Not Need<br/>O(n^2) Memory<br/>2112.05682"]
    flash["FlashAttention<br/>2205.14135"]
    gqa["GQA: Training Generalized<br/>Multi-Query Transformer Models<br/>2305.13245"]
    target["FlashAttention-2<br/>2307.08691"]

    transformer -->|helpful| target
    volta -->|helpful| target
    mqa -->|helpful| target
    reformer -->|helpful| target
    memory -->|helpful| target
    gqa -->|helpful| target
    online ==>|essential| target
    flash ==>|essential| target

    vocabulary -->|helpful| online
    sublinear -->|helpful| flash
    transformer -->|helpful| flash
    volta -->|helpful| flash
    online -->|helpful| flash
    butterfly -->|helpful| flash
    keops -->|helpful| flash
    movement -->|helpful| flash
    memory -->|helpful| flash

    classDef target fill:#172554,color:#fff,stroke:#60a5fa,stroke-width:3px
    classDef essential fill:#7f1d1d,color:#fff,stroke:#f87171,stroke-width:2px
    classDef helpful fill:#164e63,color:#fff,stroke:#22d3ee

    class target target
    class online,flash essential
    class transformer,vocabulary,sublinear,volta,mqa,reformer,butterfly,keops,movement,memory,gqa helpful
```

The run expanded the target and both essential prerequisites. It considered 77
unique candidates, made 90 inference requests, and completed without failures.
Neither expanded prerequisite produced another essential edge, so traversal
terminated after the second level.

## Response

An abridged response from the example run:

```json
{
  "target_arxiv_id": "2307.08691",
  "papers": [
    {
      "arxiv_id": "2307.08691",
      "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
    },
    {
      "arxiv_id": "2205.14135",
      "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    },
    {
      "arxiv_id": "1805.02867",
      "title": "Online normalizer calculation for softmax"
    }
  ],
  "relationships": [
    {
      "source_arxiv_id": "2205.14135",
      "target_arxiv_id": "2307.08691",
      "relation": "essential",
      "evidence": "FlashAttention-2 directly builds upon the algorithm introduced in FlashAttention, identifies its suboptimal work partitioning, and refines its computational framework.",
      "provenance": {
        "classifier": "model",
        "candidate_source": "semantic_scholar_citation_graph",
        "retrieval_depth": 1,
        "paths": [
          {
            "citations": [
              {
                "source_arxiv_id": "2205.14135",
                "target_arxiv_id": "2307.08691",
                "contexts": [
                  "FlashAttention exploits the asymmetric GPU memory hierarchy to bring significant memory saving and runtime speedup, with no approximation."
                ],
                "intents": ["background", "methodology"],
                "is_influential": true
              }
            ]
          }
        ]
      }
    }
  ],
  "generation": {
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507:nscale",
    "metrics": {
      "duration_seconds": 1146.193,
      "papers_expanded": 3,
      "unique_candidates_considered": 77,
      "inference_requests": 90,
      "checkpoint_hits": 0,
      "levels": [
        {
          "depth": 0,
          "papers_expanded": 1,
          "candidates_classified": 17,
          "essential_edges": 2,
          "helpful_edges": 6,
          "failed_classifications": 0
        },
        {
          "depth": 1,
          "papers_expanded": 2,
          "candidates_classified": 73,
          "essential_edges": 0,
          "helpful_edges": 9,
          "failed_classifications": 0
        }
      ]
    },
    "complete": true,
    "failed_candidates": []
  }
}
```

The complete response includes every selected paper, relationship, citation
context, and generation metric.

## Installation

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `HF_TOKEN` in `.env`. To reproduce the example, set:

```dotenv
MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507:nscale
```

`S2_API_KEY` is optional. `MODEL_BASE_URL` can point the same model interface at
another OpenAI-compatible endpoint.

## CLI

```powershell
.\.venv\Scripts\knowledge-maps.exe build 2307.08691
```

## API

```powershell
.\.venv\Scripts\uvicorn.exe knowledge_maps.api:create_app --factory
```

```http
POST /knowledge-maps
Content-Type: application/json

{"arxiv_id_or_url": "2307.08691"}
```

Successful model judgments are stored in
`.knowledge_maps/checkpoints.sqlite3` and reused when the model, prompt, paper
data, and citation evidence are unchanged.
