"""Coverage is a checked invariant, not a documentation promise (4.2.2).

Two failure modes matter. At spec time, a method added to the experimental
matrix but neither implemented nor declared pending is an empty cell that would
otherwise show up as a blank row in a paper table; so is a mechanism no method
exercises. At run time, a declared grid cell with no result row is an empty
cell, and a result row not in the declared grid is untraceable (4.7.4). Both
must raise, and the matrix must say what its source is.
"""

import numpy as np
import pytest

import scripts.run_benchmark as rb
from scripts.models.detector import Detector

EXPECTED_MATRIX = {
    "euclidean": ("static",),
    "mahalanobis": ("static", "covariance"),
    "isolation_forest": ("static",),
    "autoencoder": ("static", "generative"),
    "vae": ("static", "generative"),
    "markov": ("temporal",),
    "gru_predictor": ("temporal",),
    "lstm_predictor": ("temporal",),
    "lstm_vae": ("temporal", "generative"),
    "gru_mahalanobis": ("temporal", "covariance"),
}


def _config(**overrides):
    benchmark = {
        "models": ["euclidean", "pca"],
        "novelty_settings": ["leave_one_action_out", "unseen_transition"],
        "splits": ["train", "test"],
        "seeds": [42, 43],
        "context_lengths": [1, 3],
        "results_dir": "unused",
    }
    benchmark.update(overrides)
    return {"benchmark": benchmark, "random_seed": 42}


def _provider(cell, config):
    train = np.random.default_rng(cell.seed).normal(size=(32, 4))
    score = np.random.default_rng(cell.seed + 100).normal(size=(8, 4))
    return {"train": train, "score": score}


def _small_run(tmp_path, monkeypatch, **overrides):
    results = tmp_path / "results"
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42, 43],
        context_lengths=[1],
        results_dir=str(results),
        **overrides,
    )
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", {"leave_one_action_out": _provider})
    rb.run_benchmark(cfg)
    return cfg, results


# --- spec time -------------------------------------------------------------


def test_the_matrix_matches_the_cited_source():
    assert rb.MECHANISMS == ("static", "temporal", "generative", "covariance")
    assert rb.EXPERIMENTAL_MATRIX == EXPECTED_MATRIX
    assert "improvements_on_vad.md" in rb.MATRIX_SOURCE
    assert "24" in rb.MATRIX_SOURCE


def test_the_shipped_spec_is_fully_accounted_for():
    summary = rb.verify_matrix_spec()

    assert summary == {"methods": len(EXPECTED_MATRIX), "mechanisms": 4}
    accounted = set(rb.METHOD_MODELS) | set(rb.PENDING_METHODS)
    assert accounted == set(EXPECTED_MATRIX), "every method is implemented or declared pending, with no silent gap"


def test_every_implemented_method_maps_to_a_real_detector():
    for method, model_id in rb.METHOD_MODELS.items():
        assert model_id in rb.MODEL_FACTORIES, f"matrix method {method!r} maps to a missing model"
        assert isinstance(rb.build_model(model_id, 0), Detector)


def test_every_pending_method_names_its_task():
    for method, task in rb.PENDING_METHODS.items():
        assert method in EXPECTED_MATRIX
        assert task and task[0].isdigit(), f"{method!r} must cite the task that implements it"


def test_a_silently_dropped_method_is_an_empty_cell(monkeypatch):
    monkeypatch.delitem(rb.METHOD_MODELS, "euclidean")

    with pytest.raises(RuntimeError, match="euclidean"):
        rb.verify_matrix_spec()


def test_a_method_both_implemented_and_pending_is_rejected(monkeypatch):
    monkeypatch.setitem(rb.PENDING_METHODS, "euclidean", "4.3.1")

    with pytest.raises(RuntimeError, match="both"):
        rb.verify_matrix_spec()


def test_an_unknown_model_id_is_rejected(monkeypatch):
    monkeypatch.setitem(rb.METHOD_MODELS, "euclidean", "euclidian")

    with pytest.raises(ValueError, match="euclidian"):
        rb.verify_matrix_spec()


def test_a_mechanism_exercised_by_nothing_is_rejected(monkeypatch):
    for method in rb.EXPERIMENTAL_MATRIX:
        monkeypatch.setitem(rb.EXPERIMENTAL_MATRIX, method, ("static",))

    with pytest.raises(RuntimeError, match="temporal"):
        rb.verify_matrix_spec()


def test_a_method_with_no_mechanism_is_rejected(monkeypatch):
    monkeypatch.setitem(rb.EXPERIMENTAL_MATRIX, "euclidean", ())

    with pytest.raises(RuntimeError, match="no mechanism"):
        rb.verify_matrix_spec()


def test_an_unknown_mechanism_name_is_rejected(monkeypatch):
    monkeypatch.setitem(rb.EXPERIMENTAL_MATRIX, "euclidean", ("static", "magic"))

    with pytest.raises(ValueError, match="magic"):
        rb.verify_matrix_spec()


# --- run time --------------------------------------------------------------


def test_verify_coverage_passes_on_a_completed_run(tmp_path, monkeypatch):
    cfg, results = _small_run(tmp_path, monkeypatch)

    summary = rb.verify_coverage(cfg, str(results))

    assert summary == {"declared": 2, "present": 2}


def test_verify_coverage_lists_an_empty_cell(tmp_path, monkeypatch):
    cfg, results = _small_run(tmp_path, monkeypatch)
    rows = sorted(name for name in results.iterdir() if name.suffix == ".json" and name.name != rb.COVERAGE_FILENAME)
    victim = rows[0]
    victim.unlink()

    with pytest.raises(RuntimeError, match=victim.stem):
        rb.verify_coverage(cfg, str(results))


def test_verify_coverage_rejects_a_foreign_row(tmp_path, monkeypatch):
    cfg, results = _small_run(tmp_path, monkeypatch)
    (results / "leftover.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="leftover"):
        rb.verify_coverage(cfg, str(results))


def test_verify_coverage_on_a_missing_directory_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        rb.verify_coverage(_config(results_dir=str(tmp_path / "nope")), str(tmp_path / "nope"))


def test_run_benchmark_verifies_coverage_at_the_end(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()  # a cell ran but wrote nothing: the classic silent skip
    monkeypatch.setattr(rb, "run_cell", lambda cell, config, results_dir: {})
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42],
        context_lengths=[1],
        results_dir=str(results),
    )

    with pytest.raises(RuntimeError, match="empty cell"):
        rb.run_benchmark(cfg)


def test_run_benchmark_checks_the_spec_before_the_first_cell(tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("spec check first")

    monkeypatch.setattr(rb, "verify_matrix_spec", _boom)
    monkeypatch.setattr(rb, "run_cell", lambda *args: pytest.fail("a cell ran before the spec check"))

    with pytest.raises(RuntimeError, match="spec check first"):
        rb.run_benchmark(_config(results_dir=str(tmp_path / "results")))
