# MoraLink data

The Hugging Face release is at [Intellexus/MoraLink](https://huggingface.co/datasets/Intellexus/MoraLink).

## Benchmark files

- `data/fables.json`: 709 fable documents and their stable IDs.
- `data/queries.json`: 668 unique moral queries.
- `data/qrels.json`: the relevant fable IDs for every query.
- `data/clusters.json`: the final 558 moral groups.
- `data/moral_to_cluster.json`: the mapping from each moral query to its group.

The 558 groups contain 469 groups with one fable and 89 groups with more than one fable. Every fable appears exactly once in the group mapping.

## Generated summaries

- `summaries/gemini-3-flash.json`: Gemini 3 Flash summaries for all 709 fables.
- `summaries/gemma-4-e2b.json`: Gemma 4 E2B summaries for all 709 fables.
- `summaries/gemma-4-e4b.json`: Gemma 4 E4B summaries for all 709 fables.
- `summaries/gemma-4-31b.json`: Gemma 4 31B summaries for all 709 fables.

The exact prompts are in [`../prompts/summary_generation.json`](../prompts/summary_generation.json). Summaries were generated once and reused across retrieval models.

### Frozen Gemini error

The Gemini file contains 708 valid `direct_moral` summaries. The row for fable `gibbs_137_584` preserves the saved API error placeholder in its four summary fields. The reported experiments used this exact frozen file. We preserve the row so the released data matches the reported runs. We did not silently regenerate it.

## Rankings and analysis files

- `rankings/manifest.json`: metadata for the 144 official zero-shot rankings.
- `rankings/*.json`: compact top-10 results for each official run.
- `analysis/per_query_paper_metrics.csv.gz`: 96,192 per-query metric rows, covering 144 runs and 668 queries.
- `analysis/word_overlap_rank_changes.csv.gz`: 106,212 per-query rows used for the 159 word-overlap comparisons.

The per-query metric file stores the reciprocal-rank contribution needed when the first relevant fable is below rank 10. This lets users reproduce the paper-compatible MRR from compact top-10 ranking files.

## Moral-grouping records

The release also includes:

- [`annotation/PROTOCOL.md`](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/PROTOCOL.md): the grouping procedure.
- [`annotation/DECISION_LOG.csv`](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/DECISION_LOG.csv): preserved accepted and rejected decisions.
- [`annotation/MORAL_EDITS.csv`](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/MORAL_EDITS.csv): 23 edited morals with their earlier text.
- [`annotation/AUTHOR_CONFIRMATIONS.json`](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/AUTHOR_CONFIRMATIONS.json): two final decisions whose earlier prose reasons were not preserved.

See [the moral-grouping guide](MORAL_GROUPING.md) for a short explanation.

## Integrity checks

[`../configs/artifacts.yaml`](../configs/artifacts.yaml) records the expected paths, counts, and SHA-256 hashes. Run `scripts/validate_data.py` after download. Do not use a partially downloaded directory for experiments.
