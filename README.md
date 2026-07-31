# Knowledge Maps

Knowledge Maps accepts an arXiv ID or URL and returns a directed prerequisite
graph as JSON.

## Pipeline

1. Fetch the target paper's metadata and abstract from arXiv.
2. Retrieve direct references and citation evidence from Semantic Scholar.
3. Classify each candidate as `essential`, `helpful`, `related_only`, or
   `not_relevant`.
4. Expand one additional citation hop through `essential` and `helpful` direct
   candidates.
5. Return selected papers, prerequisite relationships, model evidence, citation
   paths, and generation metadata.

Each relationship points from a prerequisite paper to the requested paper.

## Example

Output for
[*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150):

```mermaid
flowchart LR
    subgraph second[Discovered two citation hops back]
        goodman["A Bit of Progress in<br/>Language Modeling<br/>cs/0108005"]
        rnn["Difficulty of Training RNNs<br/>1211.5063"]
        encdec["Properties of Neural MT<br/>1409.1259"]
        conv["Convolutional Seq2Seq<br/>1705.03122"]
        summarization["Neural Attention for<br/>Summarization<br/>1509.00685"]
        supervised["Supervised Attentions<br/>1608.00112"]
        nonauto["Non-Autoregressive NMT<br/>1711.02281"]
    end

    subgraph direct[Directly cited candidates]
        billion["One Billion Word Benchmark<br/>1312.3005"]
        bahdanau["Neural Machine Translation with<br/>Attention<br/>1409.0473"]
        transformer["Attention Is All You Need<br/>1706.03762"]
        wiki["Generating Wikipedia by<br/>Summarizing Long Sequences<br/>1801.10198"]
        average["Average Attention Network<br/>1805.00631"]
    end

    target["Fast Transformer Decoding:<br/>One Write-Head is All You Need<br/>1911.02150"]

    bahdanau ==>|essential| target
    transformer ==>|essential| target
    billion -->|helpful| target
    wiki -->|helpful| target
    average -->|helpful| target
    goodman -->|helpful| target
    rnn -->|helpful| target
    encdec -->|helpful| target
    conv -->|helpful| target
    summarization -->|helpful| target
    supervised -->|helpful| target
    nonauto -->|helpful| target

    classDef target fill:#172554,color:#fff,stroke:#60a5fa,stroke-width:3px
    classDef essential fill:#7f1d1d,color:#fff,stroke:#f87171,stroke-width:2px
    classDef helpful fill:#164e63,color:#fff,stroke:#22d3ee

    class target target
    class bahdanau,transformer essential
    class billion,wiki,average,goodman,rnn,encdec,conv,summarization,supervised,nonauto helpful
```

This run returned 13 paper nodes, 12 relationships, and no failed candidates.

## Response

```json
{
  "target_arxiv_id": "1911.02150",
  "papers": [
    {
      "arxiv_id": "1911.02150",
      "title": "Fast Transformer Decoding: One Write-Head is All You Need"
    },
    {
      "arxiv_id": "1706.03762",
      "title": "Attention Is All You Need"
    }
  ],
  "relationships": [
    {
      "source_arxiv_id": "1706.03762",
      "target_arxiv_id": "1911.02150",
      "relation": "essential",
      "evidence": "The target paper directly references and builds upon the Transformer architecture introduced in 'Attention Is All You Need'...",
      "provenance": {
        "classifier": "model",
        "candidate_source": "semantic_scholar_citation_graph",
        "retrieval_depth": 1,
        "paths": [
          {
            "citations": [
              {
                "source_arxiv_id": "1706.03762",
                "target_arxiv_id": "1911.02150",
                "contexts": [
                  "We introduce multi-query Attention as a variation of multi-head attention as described in [Vaswani et al., 2017]."
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
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "complete": true,
    "failed_candidates": []
  }
}
```

The response includes complete paper metadata. The example above contains only
the fields needed to show the graph contract and one relationship's provenance.

## Installation

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `HF_TOKEN` in `.env`. `S2_API_KEY` is optional.

## CLI

```powershell
.\.venv\Scripts\knowledge-maps.exe build 1911.02150
```

## API

```powershell
.\.venv\Scripts\uvicorn.exe knowledge_maps.api:create_app --factory
```

```http
POST /knowledge-maps
Content-Type: application/json

{"arxiv_id_or_url": "1911.02150"}
```

Successful model judgments are stored in
`.knowledge_maps/checkpoints.sqlite3` and reused when the model, prompt, paper
data, and citation evidence are unchanged.
