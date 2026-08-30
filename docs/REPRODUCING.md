# Reproducing MoraLink results

This guide separates quick checks from expensive model runs.

## Quick checks

Install the project and its test tools:

```bash
uv sync --extra analysis --extra dev
uv run ruff check .
uv run pytest
```

Download and validate the frozen data:

```bash
uv run python scripts/download_data.py --output-dir artifacts/moralink
uv run python scripts/validate_data.py --root artifacts/moralink
```

Create the four released result tables without rerunning models:

```bash
uv run python scripts/create_paper_tables.py \
  --results-dir results \
  --output-dir outputs/tables
```

## Rebuild the one-versus-several analysis

This analysis uses the frozen per-query metric contributions:

```bash
uv run python scripts/analyze_relevance_groups.py \
  --data-dir artifacts/moralink \
  --per-query-metrics artifacts/moralink/analysis/per_query_paper_metrics.csv.gz \
  --official-runs configs/official_runs.yaml \
  --output outputs/relevance_groups.csv
```

The output has 288 rows: 144 retrieval runs, each evaluated on queries with one relevant fable and queries with several relevant fables.

## Rebuild the word-overlap analysis

```bash
uv run python scripts/analyze_word_overlap.py \
  --data-dir artifacts/moralink \
  --rank-changes artifacts/moralink/analysis/word_overlap_rank_changes.csv.gz \
  --official-runs configs/official_runs.yaml \
  --output-dir outputs/word_overlap
```

This produces 159 matched comparisons. Each comparison keeps the query, retrieval model, and instruction setting fixed, and changes only the candidate text from the original fable to a generated summary.

## Rerun zero-shot retrieval

One example is:

```bash
uv run python scripts/run_zero_shot.py \
  --data-dir artifacts/moralink \
  --model all-MiniLM-L6-v2 \
  --representation fable_only \
  --instruction no_instr \
  --summary-source gemini \
  --batch-size 64 \
  --output outputs/minilm_fable_only.json
```

The 144 official settings are listed in [`../configs/official_runs.yaml`](../configs/official_runs.yaml). Running all of them downloads 24 embedding models and requires substantial compute.

## Run BM25

BM25 uses the same benchmark and document representations as the embedding runs:

```bash
uv run python scripts/run_bm25.py \
  --data-dir artifacts/moralink \
  --representation fable_only \
  --summary-source gemini \
  --output outputs/bm25_fable_only.json
```

## Evaluate a saved ranking

Use paper mode to reproduce the metric definition used in the reported tables:

```bash
uv run python scripts/evaluate_rankings.py \
  --data-dir artifacts/moralink \
  --ranking outputs/minilm_fable_only.json \
  --metric-mode paper \
  --k 10
```

Use `--metric-mode standard` for conventional MRR@10, which assigns zero when the first relevant fable appears below rank 10.

## Check a fine-tuning run before using a GPU

The dry run loads the training data, checks exact counts, and prints the train and validation sizes without loading a model:

```bash
uv sync --extra training
```

```bash
uv run python scripts/run_fine_tuning.py \
  --config configs/fine_tuning/curriculum.yaml \
  --model MiniLM-L6 \
  --seed 42 \
  --storal-path /path/to/storal_pairs_clean_no_morables_overlap.json \
  --tf1-path /path/to/tf1_synthetic_low_iou_no_morables_overlap \
  --output outputs/curriculum_minilm \
  --dry-run
```

Remove `--dry-run` to train. Full fine-tuning needs the optional training dependencies and the external STORAL and TF1 datasets. These third-party training datasets are not redistributed here.

## Direct LLM ranking

Direct LLM ranking is optional and uses a paid provider API. Install the optional provider clients first:

```bash
uv sync --extra llm
```

Use `--dry-run` to write one complete prompt without reading an API key or making a request:

```bash
uv run python scripts/run_llm_ranking.py \
  --provider openai \
  --model gpt-5.4-nano \
  --data-dir artifacts/moralink \
  --representation fable_only \
  --output outputs/direct_llm_prompt.txt \
  --dry-run
```

Remove `--dry-run` and provide the matching API key through the environment to run all queries. The fixed summary-based embedding results do not require any API access.
