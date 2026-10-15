"""First-order Markov novelty detector (3.4.2).

Models normal behaviour as a discrete chain over quantised behavioural states and
scores each observation by the negative log-likelihood of the transition that
produced it:

    s_t = -log P(b_t | b_{t-1}),   s_0 = -log P(b_0)

`improvements_on_vad.md:281` includes this as the cheapest temporal baseline —
"is a rare transition enough, without features?". The fitted artifact is exactly
the transition matrix `P(b_t | b_{t-1})` (plus the start prior), so Phase 4 can
resume and audit it like every other detector.
"""

import numpy as np

from scripts.models.non_generative.base import FittedModel, register

DEFAULT_SMOOTHING = 1.0


@register
class MarkovDetector(FittedModel):
    kind = "markov"
    estimator = "first_order"

    def __init__(self, n_states=None, smoothing=DEFAULT_SMOOTHING):
        if n_states is not None and n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        if smoothing <= 0:
            raise ValueError(f"smoothing must be > 0, got {smoothing}")
        self.n_states = n_states
        self.smoothing = float(smoothing)
        self.transition_matrix_ = None
        self.prior_ = None

    def _fit_impl(self, sequences):
        sequences = _as_sequences(sequences)
        observed = int(sequences.max()) + 1
        n_states = observed if self.n_states is None else int(self.n_states)
        if observed > n_states:
            raise ValueError(f"data contains state {observed - 1} but n_states={n_states}")

        transitions = np.full((n_states, n_states), self.smoothing, dtype=np.float64)
        starts = np.full(n_states, self.smoothing, dtype=np.float64)
        for sequence in sequences:
            starts[sequence[0]] += 1
            transitions[sequence[:-1], sequence[1:]] += 1

        self.n_states = n_states
        self.transition_matrix_ = transitions / transitions.sum(axis=1, keepdims=True)
        self.prior_ = starts / starts.sum()

    def score(self, sequences):
        self._require_fitted()
        array = _as_sequences(sequences)
        if int(array.max()) >= self.n_states:
            raise ValueError(f"data contains state {int(array.max())} but the chain has {self.n_states} states")
        log_transition = np.log(self.transition_matrix_)
        scores = log_transition[array[:, :-1], array[:, 1:]]
        first = -np.log(self.prior_)[array[:, 0]]
        return -np.concatenate([first[:, None], scores], axis=1)

    def state_dict(self):
        self._require_fitted()
        return {"transition_matrix": np.asarray(self.transition_matrix_), "prior": np.asarray(self.prior_)}

    @classmethod
    def from_state(cls, state, *, config, estimator):
        config = config or {}
        for key in ("transition_matrix", "prior"):
            if key not in state:
                raise ValueError(f"markov artifact missing state entry: {key}")
        model = cls(n_states=None, smoothing=config.get("smoothing", DEFAULT_SMOOTHING))
        model.transition_matrix_ = np.asarray(state["transition_matrix"], dtype=np.float64)
        model.prior_ = np.asarray(state["prior"], dtype=np.float64)
        model.n_states = model.transition_matrix_.shape[0]
        model._require_fitted()
        return model

    def _require_fitted(self):
        if self.transition_matrix_ is None or self.prior_ is None:
            raise RuntimeError("MarkovDetector used before fit; call fit() or load a checkpoint")
        if self.transition_matrix_.shape[0] != self.transition_matrix_.shape[1]:
            raise ValueError(f"transition matrix must be square, got {self.transition_matrix_.shape}")
        if self.prior_.shape != (self.transition_matrix_.shape[0],):
            raise ValueError(
                f"prior shape {self.prior_.shape} does not match the {self.transition_matrix_.shape[0]}-state chain"
            )


def _as_sequences(sequences):
    """Coerce to an integer `(n_sequences, length)` block of state indices."""
    array = np.asarray(sequences)
    if array.size == 0:
        raise ValueError("MarkovDetector received no states")
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError("Markov states must be integer-coded; got a non-integral float array")
        array = array.astype(np.int64)
    elif array.dtype.kind not in ("i", "u", "b"):
        raise ValueError(f"Markov states must be integer-coded, got dtype {array.dtype}")
    else:
        array = array.astype(np.int64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected a (n_sequences, length) array of states, got shape {array.shape}")
    if np.any(array < 0):
        raise ValueError("Markov states must be non-negative")
    return array
