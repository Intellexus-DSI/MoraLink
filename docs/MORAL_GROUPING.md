# How the moral groups were built

MoraLink starts with 709 fables from MORABLES. Some fables express the same lesson. Grouping them prevents the benchmark from marking a correct retrieval as wrong only because a different fable expresses the same moral.

## Who made the decisions

Three annotators worked together: one PhD researcher, one PhD candidate, and one graduate student. All three have academic research backgrounds and are proficient in English.

They discussed candidates together and produced one agreed grouping. They did not create three independent label sets, so independent agreement cannot be calculated after the fact. The final grouping was not independently validated by a separate annotator.

## How candidates were found

The team used four ways to find possible groups:

1. Exact duplicate moral text.
2. Similar moral pairs found with sentence embeddings.
3. Possible merges and splits proposed by Claude Sonnet 4.5.
4. Human reading of remaining single morals.

These tools only proposed candidates. They did not make the final decisions.

## How groups were decided

The annotators asked whether two morals state the same underlying lesson. Sharing a topic or a few words was not enough.

A proposed group could be accepted, split into smaller groups, expanded, rejected, or moved into a better group. The annotators discussed unclear cases until they reached one shared decision.

The final benchmark has 558 groups: 469 groups contain one fable and 89 contain more than one fable. It has 668 unique moral queries and covers all 709 fables exactly once.

## Moral edits

During review, 23 moral texts were corrected or clarified. The release keeps both the earlier and final wording. The edits include spelling and punctuation fixes, small wording corrections, and a small number of changes made to better match the fable or accepted group.

## Full records

- [Grouping protocol](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/PROTOCOL.md)
- [Preserved grouping decisions](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/DECISION_LOG.csv)
- [All 23 moral edits](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/MORAL_EDITS.csv)
- [Author confirmations](https://huggingface.co/datasets/Intellexus/MoraLink/blob/main/annotation/AUTHOR_CONFIRMATIONS.json)

The frozen group mapping is the final source of truth. The decision log preserves the evidence that still exists, but it is not a complete word-for-word transcript of every team discussion.
