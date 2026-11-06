"""Cosine scores against class prototypes; a zero prototype row scores -inf, so it never wins. K1 enrolment and the
C1 few-shot curve that use them are in notebook 11."""
import numpy as np

from serpentguard.run_outputs import softmax


def unit(vectors):
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def masked_cosine(embeddings, prototypes):
    """(N, C) cosine of each embedding to each prototype; a zero prototype row scores -inf."""
    embeddings, prototypes = unit(embeddings), np.asarray(prototypes, dtype=float)
    norms = np.linalg.norm(prototypes, axis=1)
    active = norms > 0
    scores = np.full((len(embeddings), len(prototypes)), -np.inf)
    scores[:, active] = embeddings @ (prototypes[active] / norms[active, None]).T
    return scores


def cosine_probs(embeddings, prototypes, scale):
    """softmax(scale x cosine) over the classes that have a prototype; the others get 0."""
    return softmax(scale * masked_cosine(embeddings, prototypes))
