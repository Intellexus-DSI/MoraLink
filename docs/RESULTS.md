# MoraLink results

The [`../results/`](../results) directory contains the exact camera-ready result tables before display rounding.

## Files

- [`zero_shot.csv`](../results/zero_shot.csv): 144 runs from 24 models, two query settings, and three document representations.
- [`fine_tuning.csv`](../results/fine_tuning.csv): 36 aggregate, change, and per-model rows for the four reported training strategies.
- [`relevance_groups.csv`](../results/relevance_groups.csv): 288 rows that separate queries with one relevant fable from queries with several.
- [`word_overlap.csv`](../results/word_overlap.csv): 159 matched query-summary comparisons.

Generate Markdown and LaTeX views with `scripts/create_paper_tables.py`. Values are rounded to three decimals only in those views. The CSV files keep their original precision.

## Paper-compatible MRR and standard MRR@10

The paper-compatible MRR uses the rank of the first relevant fable in the full 709-document ordering. MAP@10 and Recall@10 stop at rank 10. The paper reports this reciprocal-rank value under the label MRR@10, so the release keeps that label when reproducing the paper.

The standard MRR@10 option stops at rank 10. If the first relevant fable is at rank 11, the two values differ:

- paper-compatible MRR: `1 / 11`
- standard MRR@10: `0`

This difference is tested directly in [`test_metrics.py`](../tests/test_metrics.py), including the rank 11 case. Use `scripts/evaluate_rankings.py --metric-mode paper` to match the released paper tables. Use `--metric-mode standard` for the conventional cutoff-at-10 definition.

## Main reading of the results

Retrieving original fables directly from their morals remains difficult. Generated moral-style summaries improve retrieval because they give the retriever a representation closer to the query. The comparison with original human-written morals shows that the generated summaries still leave a large gap. Fine-tuning also improves results, especially when real moral-story examples are used.

For more commands, see [the reproduction guide](REPRODUCING.md).
