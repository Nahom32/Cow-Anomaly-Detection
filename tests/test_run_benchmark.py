"""Benchmark grid enumeration and one-row-per-run persistence (4.2.1).

What has to hold before a ~225-cell grid is launched: axes are validated
*before* execution (a typo must not survive to row 170), every executed cell
writes exactly one JSON row with the declared schema into a single results
directory (4.7.4 traceability), a missing novelty provider fails naming the
Phase 6 task that will supply it rather than scoring a different setting than
the row claims, and score/fit integrity failures raise instead of becoming a
plausible-looking metric.
"""

import json
import os

import numpy as np
import pytest

import scripts.run_benchmark as rb
from scripts.models.detector import Detector
from scripts.utils.hashing import hash_array


def _config(**overrides):
    """A small explicit grid: 2 x 2 x 2 x 2 x 2 = 32 cells."""
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


def _providers(**mapping):
    return dict(mapping)


def test_grid_is_the_full_declared_cross_product():
    cells = rb.benchmark_grid(_config())

    assert len(cells) == 32
    assert cells[0] == rb.BenchmarkCell("euclidean", "leave_one_action_out", "train", 42, 1)
    assert cells[-1] == rb.BenchmarkCell("pca", "unseen_transition", "test", 43, 3)
    ids = [cell.cell_id() for cell in cells]
    assert len(set(ids)) == 32, "cell ids name the row files, so they must be unique"
    assert set(cells[0].axes()) == {"model", "novelty_setting", "split", "seed", "context_length"}


def test_the_declared_config_grid_is_constructible():
    from scripts.config import CONFIG

    cells = rb.benchmark_grid(CONFIG)

    assert cells, "the shipped CONFIG must declare a runnable grid (4.2.1)"
    declared = CONFIG["benchmark"]
    expected = (
        len(declared["models"])
        * len(declared["novelty_settings"])
        * len(declared["splits"])
        * len(declared["seeds"])
        * len(declared["context_lengths"])
    )
    assert len(cells) == expected


@pytest.mark.parametrize("axis", rb.GRID_AXES)
def test_an_empty_axis_is_rejected_before_execution(axis):
    overrides = {axis: []}
    with pytest.raises(ValueError, match=axis):
        rb.benchmark_grid(_config(**overrides))


def test_an_unknown_model_or_setting_is_rejected_at_grid_time():
    with pytest.raises(ValueError, match="unknown value 'euclidian'"):
        rb.benchmark_grid(_config(models=["euclidian"]))
    with pytest.raises(ValueError, match="unknown value 'held_ouy_behaviour'"):
        rb.benchmark_grid(_config(novelty_settings=["held_ouy_behaviour"]))
    with pytest.raises(ValueError, match="unknown value 'trani'"):
        rb.benchmark_grid(_config(splits=["trani"]))


def test_a_non_integer_seed_or_zero_context_is_rejected():
    with pytest.raises(ValueError, match="seeds"):
        rb.benchmark_grid(_config(seeds=["42"]))
    with pytest.raises(ValueError, match="context lengths must be >= 1"):
        rb.benchmark_grid(_config(context_lengths=[0]))


def test_grid_time_validation_names_the_missing_benchmark_section():
    with pytest.raises(ValueError, match="'benchmark'"):
        rb.benchmark_grid({"random_seed": 42})


@pytest.mark.parametrize("setting,task", sorted(rb.NOVELTY_SETTINGS.items()))
def test_a_missing_provider_fails_naming_its_phase_6_task(setting, task, tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers())
    cell = rb.BenchmarkCell("euclidean", setting, "test", 42, 1)

    with pytest.raises(RuntimeError, match=task):
        rb.run_cell(cell, _config(results_dir=str(tmp_path)))


def test_run_writes_one_row_per_cell_into_one_results_dir(tmp_path, monkeypatch):
    results = tmp_path / "results"
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42, 43],
        context_lengths=[1],
        results_dir=str(results),
    )
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=_provider))

    rows = rb.run_benchmark(cfg)

    assert len(rows) == 2
    assert sorted(p.name for p in results.iterdir()) == [
        "coverage.json",
        "euclidean__leave_one_action_out__test__s42__k1.json",
        "euclidean__leave_one_action_out__test__s43__k1.json",
        "results.csv",
    ]
    for row in rows:
        assert list(row) == list(rb.RESULT_FIELDS)
        # The fingerprint is the hash of the exact rows the provider returned,
        # so an auditor can re-derive it from the train rows (4.1.2 / 4.7.4).
        expected_train = np.random.default_rng(row["seed"]).normal(size=(32, 4))
        assert row["fit_fingerprint"] == hash_array(expected_train)
        assert row["n_train"] == 32
        assert row["n_scored"] == 8
        assert row["config_hash"]
        assert row["created_at"].endswith("Z")

    with open(results / "results.csv", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0].split(",") == list(rb.RESULT_FIELDS)
    assert len(lines) == 3, "header + one row per executed cell"
    assert {line.split(",")[3] for line in lines[1:]} == {"42", "43"}


def test_results_csv_is_a_projection_regenerated_from_the_row_jsons(tmp_path, monkeypatch):
    results = tmp_path / "results"
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42],
        context_lengths=[1],
        results_dir=str(results),
    )
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=_provider))
    rb.run_benchmark(cfg)
    original = (results / "results.csv").read_text(encoding="utf-8")

    (results / "results.csv").unlink()
    regenerated = rb.write_results_csv(str(results))

    assert open(regenerated, encoding="utf-8").read() == original


def test_a_provider_must_return_both_row_sets(tmp_path, monkeypatch):
    cell = rb.BenchmarkCell("euclidean", "leave_one_action_out", "test", 42, 1)
    cfg = _config(results_dir=str(tmp_path))
    monkeypatch.setattr(
        rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=lambda cell, cfg: {"train": np.ones((4, 2))})
    )
    with pytest.raises(ValueError, match="missing 'score'"):
        rb.run_cell(cell, cfg)

    monkeypatch.setattr(
        rb,
        "SETTING_PROVIDERS",
        _providers(leave_one_action_out=lambda cell, cfg: {"train": np.empty((0, 2)), "score": np.ones((4, 2))}),
    )
    with pytest.raises(ValueError, match="no 'train' rows"):
        rb.run_cell(cell, cfg)


class _WrongLengthScores(Detector):
    def _fit_impl(self, train):
        self.n_fit_rows_ = len(train)

    def score(self, data):
        return np.zeros(len(data) - 1)


class _NonFiniteScores(Detector):
    def _fit_impl(self, train):
        self.n_fit_rows_ = len(train)

    def score(self, data):
        return np.full(len(data), np.nan)


@pytest.mark.parametrize(
    "model_cls,match",
    [(_WrongLengthScores, "scores for"), (_NonFiniteScores, "non-finite")],
    ids=["wrong_length", "non_finite"],
)
def test_score_integrity_failures_raise_instead_of_becoming_metrics(model_cls, match, tmp_path, monkeypatch):
    name = model_cls.__name__.lower()
    monkeypatch.setitem(rb.MODEL_FACTORIES, name, lambda seed: model_cls())
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=_provider))
    cell = rb.BenchmarkCell(name, "leave_one_action_out", "test", 42, 1)

    with pytest.raises(ValueError, match=match):
        rb.run_cell(cell, _config(results_dir=str(tmp_path)))


def test_build_model_hands_over_the_cell_seed(monkeypatch):
    seen = []
    monkeypatch.setitem(rb.MODEL_FACTORIES, "seed_probe", lambda seed: seen.append(seed) or _NonFiniteScores())

    rb.build_model("seed_probe", 7)

    assert seen == [7]
    with pytest.raises(ValueError, match="unknown benchmark model"):
        rb.build_model("euclidian", 7)


def test_build_model_refuses_a_non_detector(monkeypatch):
    monkeypatch.setitem(rb.MODEL_FACTORIES, "not_a_detector", lambda seed: object())

    with pytest.raises(TypeError, match="Detector"):
        rb.build_model("not_a_detector", 7)


def test_register_setting_validates_the_name_and_refuses_replacement(monkeypatch):
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers())

    with pytest.raises(ValueError, match="unknown novelty setting"):
        rb.register_setting("held_ouy_behaviour", _provider)
    rb.register_setting("leave_one_action_out", _provider)
    with pytest.raises(RuntimeError, match="already has a provider"):
        rb.register_setting("leave_one_action_out", _provider)


def test_a_cell_seeds_from_its_seed_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=_provider))
    cell = rb.BenchmarkCell("euclidean", "leave_one_action_out", "test", 43, 1)

    rb.run_cell(cell, _config(results_dir=str(tmp_path)))

    drawn = np.random.random()
    np.random.seed(43)
    assert drawn == np.random.random(), "a cell must seed from its own seed, not the global default"


def test_dry_run_prints_the_grid_and_executes_nothing(tmp_path, capsys, monkeypatch):
    def _explode(cell, config):
        raise AssertionError("dry run must not execute a cell")

    settings = {setting: _explode for setting in rb.NOVELTY_SETTINGS}
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", settings)
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42],
        context_lengths=[1],
        results_dir=str(tmp_path / "results"),
    )

    cells = rb.main(["--dry-run"], config=cfg)

    out = capsys.readouterr().out
    assert len(cells) == 1
    assert "euclidean__leave_one_action_out__test__s42__k1" in out
    assert "1 cells" in out
    assert not (tmp_path / "results").exists()


def test_row_files_are_written_atomically(tmp_path, monkeypatch):
    results = tmp_path / "results"
    cfg = _config(
        models=["euclidean"],
        novelty_settings=["leave_one_action_out"],
        splits=["test"],
        seeds=[42],
        context_lengths=[1],
        results_dir=str(results),
    )
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", _providers(leave_one_action_out=_provider))

    rb.run_benchmark(cfg)

    leftovers = [name for name in os.listdir(results) if not name.endswith((".json", ".csv"))]
    assert leftovers == [], f"atomic writers must leave no temp files behind: {leftovers}"
    with open(results / "euclidean__leave_one_action_out__test__s42__k1.json", encoding="utf-8") as handle:
        assert json.load(handle)["model"] == "euclidean"
