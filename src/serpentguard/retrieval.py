"""Rank fusion for the extractive first-aid search."""

from collections import defaultdict
from math import fsum


def reciprocal_rank_fusion(rankings, *, k=60):
    """Fuse ordered passage IDs. A passage contributes once per ranker.

    Missing IDs contribute zero. Break equal scores by stable passage ID.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    weights = defaultdict(list)
    for ranking in rankings:
        for rank, passage_id in enumerate(dict.fromkeys(ranking), 1):
            weights[passage_id].append(1 / (k + rank))
    scores = {passage: fsum(weight) for passage, weight in weights.items()}
    return sorted(scores, key=lambda passage: (-scores[passage], passage))
