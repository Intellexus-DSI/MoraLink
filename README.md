# MoraLink

MoraLink is a benchmark for retrieving a fable from a moral lesson. The released benchmark contains 709 fables, 668 moral queries, and 558 moral groups. It tests whether a retrieval model can connect a short abstract lesson to the longer narrative that expresses it.

## Paper and artifacts

- Paper: [MoraLink: Bridging Morals and Narrative Fables for Retrieval](https://openreview.net/forum?id=wpUUxOAzZY), EMNLP 2026
- Code: [Intellexus-DSI/MoraLink](https://github.com/Intellexus-DSI/MoraLink)
- Data and frozen experiment artifacts: [Intellexus/MoraLink](https://huggingface.co/datasets/Intellexus/MoraLink)

## Install

MoraLink requires Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Intellexus-DSI/MoraLink.git
cd MoraLink
uv sync --extra analysis --extra dev
```

## Download and check the data

```bash
uv run python scripts/download_data.py --output-dir artifacts/moralink
uv run python scripts/validate_data.py --root artifacts/moralink
```

The validator checks file hashes and the expected counts before an experiment starts. See [the data guide](docs/DATA.md) for the file layout.

## Run one zero-shot experiment

This example evaluates MiniLM on the original fables without a query instruction:

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

Model weights are downloaded from their original providers and are not included in this repository.

## Recreate the released tables

The exact camera-ready results are stored as CSV files in [`results/`](results). Generate readable Markdown and LaTeX versions with:

```bash
uv run python scripts/create_paper_tables.py \
  --results-dir results \
  --output-dir outputs/tables
```

This step is fast and does not rerun any model. For complete analysis and rerun commands, see [the reproduction guide](docs/REPRODUCING.md). Metric definitions are explained in [the results guide](docs/RESULTS.md).

## Main results

Retrieving original fables directly from their morals remains difficult. Generated moral-style summaries and fine-tuning improve retrieval, but a large gap remains between generated summaries and the original human-written morals. These results suggest that representing the narrative at the right level of abstraction is a major part of the challenge.

## Repository map

- [`configs/`](configs): model, instruction, artifact, and fine-tuning settings
- [`prompts/`](prompts): exact summary-generation and direct-ranking prompts
- [`scripts/`](scripts): commands for download, validation, experiments, analyses, and tables
- [`src/moralink/`](src/moralink): reusable data, retrieval, metric, analysis, and training code
- [`results/`](results): small machine-readable camera-ready result tables
- [`docs/`](docs): data, grouping, reproduction, results, and licensing guides
- [`tests/`](tests): tests for the public release

## Licenses

The code, prompts, configurations, and original documentation are released under Apache-2.0. The MoraLink benchmark data and generated summaries are derived from MORABLES and are released under CC BY-NC 4.0. Third-party models and training datasets keep their own licenses. See [the licensing guide](docs/LICENSING.md) and [`NOTICE`](NOTICE).

## Citation

MoraLink was created by Lior Livyatan, Kai Golan Hashiloni, Asif Amar, Roei Rahamim, and Kfir Bar. Citation metadata is available in [`CITATION.cff`](CITATION.cff). If you use MoraLink, please cite the paper and the MORABLES source dataset.

## Acknowledgments

This work was supported by Intellexus.
