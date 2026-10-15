"""Per-cell coverage status: complete / missing / failed (4.2.3).

A results directory only ever showed the runs that succeeded. A cell that never
ran and a cell that raised both looked like absence, so a partial run could be
read as a full one. The coverage index records each declared cell's status, read
back from the files on disk (a row JSON is `complete`, a failure marker is
`failed`, neither is `missing`), as a projection — never a second source of
truth — and `run_benchmark` fails the run after recording it.
"""

import json

import numpy as np
import pytest

import scripts.run_benchmark as rb


def _config(**overrides):
    benchmark = {
        "models": ["euclidean"],
        "novelty_settings": ["leave_one_action_out"],
        "splits": ["test"],
        "seeds": [42, 43],
        "context_lengths": [1],
        "results_dir": "unused",
    }
    benchmark.update(overrides)
    return {"benchmark": benchmark, "random_seed": 42}


def _provider(cell, config):
    train = np.random.default_rng(cell.seed).normal(size=(32, 4))
    score = np.random.default_rng(cell.seed + 100).normal(size=(8, 4))
    return {"train": train, "score": score}


def _cfg(tmp_path, **overrides):
    return _config(results_dir=str(tmp_path / "results"), **overrides)


def _run(tmp_path, monkeypatch, **overrides):
    cfg = _cfg(tmp_path, **overrides)
    monkeypatch.setattr(rb, "SETTING_PROVIDERS", {"leave_one_action_out": _provider})
    rb.run_benchmark(cfg)
    return cfg


def _index(results):
    return json.loads((results / rb.COVERAGE_FILENAME).read_text(encoding="utf-8"))


def test_status_is_complete_for_every_executed_cell(tmp_path, monkeypatch):
    cfg = _run(tmp_path, monkeypatch)

    entries = rb.coverage_status(cfg)

    assert [entry["status"] for entry in entries] == ["complete", "complete"]
    assert [entry["cell_id"] for entry in entries] == sorted(entry["cell_id"] for entry in entries)
    assert all(entry["error"] is None for entry in entries)


def test_status_is_missing_when_a_row_is_absent(tmp_path, monkeypatch):
    cfg = _run(tmp_path, monkeypatch)
    results = tmp_path / "results"
    victim = next(name for name in results.iterdir() if name.name.startswith("euclidean__"))
    victim.unlink()

    statuses = {entry["cell_id"]: entry["status"] for entry in rb.coverage_status(cfg)}

    assert statuses[victim.stem] == "missing"
    assert sorted(statuses.values()) == ["complete", "missing"]


def test_status_is_failed_when_only_a_marker_exists(tmp_path, monkeypatch):
    cfg = _run(tmp_path, monkeypatch)
    results = tmp_path / "results"
    cell = rb.benchmark_grid(cfg)[1]
    (results / f"{cell.cell_id()}.json").unlink()
    rb.record_cell_failure(cell, RuntimeError("boom"), str(results))

    entry = next(e for e in rb.coverage_status(cfg) if e["cell_id"] == cell.cell_id())

    assert entry["status"] == "failed"
    assert entry["error"] == "RuntimeError: boom"


def test_the_index_records_counts_and_the_per_cell_status(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)

    index = _index(tmp_path / "results")

    assert index["counts"] == {"complete": 2, "missing": 0, "failed": 0}
    assert {entry["status"] for entry in index["cells"]} == {"complete"}
    assert {"cell_id", "status", "error", "model", "split", "seed", "context_length"} <= set(index["cells"][0])


def test_a_failed_cell_is_recorded_and_fails_the_run(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    real = rb.run_cell

    def flaky(cell, config, results_dir=None):
        if cell.seed == 43:
            raise RuntimeError("boom")
        return real(cell, config, results_dir)

    monkeypatch.setattr(rb, "SETTING_PROVIDERS", {"leave_one_action_out": _provider})
    monkeypatch.setattr(rb, "run_cell", flaky)

    with pytest.raises(RuntimeError, match="failed"):
        rb.run_benchmark(cfg)

    results = tmp_path / "results"
    index = _index(results)
    assert index["counts"] == {"complete": 1, "missing": 0, "failed": 1}
    marker = results / f"euclidean__leave_one_action_out__test__s43__k1{rb.FAILED_SUFFIX}"
    assert json.loads(marker.read_text(encoding="utf-8"))["error"] == "RuntimeError: boom"


def test_a_failing_cell_records_a_marker_but_not_a_row(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def explode(cell, config, results_dir=None):
        raise KeyError("x")

    monkeypatch.setattr(rb, "SETTING_PROVIDERS", {"leave_one_action_out": _provider})
    monkeypatch.setattr(rb, "run_cell", explode)

    with pytest.raises(RuntimeError, match="failed"):
        rb.run_benchmark(cfg)

    index = _index(tmp_path / "results")
    assert index["counts"] == {"complete": 0, "missing": 0, "failed": 2}


def test_failed_markers_and_the_index_are_not_rows(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, seeds=[42])

    def explode(cell, config, results_dir=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(rb, "SETTING_PROVIDERS", {"leave_one_action_out": _provider})
    monkeypatch.setattr(rb, "run_cell", explode)
    with pytest.raises(RuntimeError):
        rb.run_benchmark(cfg)

    csv = (tmp_path / "results" / "results.csv").read_text(encoding="utf-8")
    assert csv.splitlines() == [",".join(rb.RESULT_FIELDS)], "a failed cell contributes no results row"


def test_a_failed_cell_is_an_empty_cell_for_verify_coverage(tmp_path, monkeypatch):
    cfg = _run(tmp_path, monkeypatch)
    results = tmp_path / "results"
    cell = rb.benchmark_grid(cfg)[0]
    (results / f"{cell.cell_id()}.json").unlink()
    rb.record_cell_failure(cell, RuntimeError("boom"), str(results))

    with pytest.raises(RuntimeError, match="empty cell"):
        rb.verify_coverage(cfg, str(results))


def test_the_index_is_regenerated_not_appended(tmp_path, monkeypatch):
    cfg = _run(tmp_path, monkeypatch)
    results = tmp_path / "results"
    victim = next(name for name in results.iterdir() if name.name.startswith("euclidean__"))
    victim.unlink()

    rb.write_coverage_index(cfg, str(results))

    assert _index(results)["counts"] == {"complete": 1, "missing": 1, "failed": 0}


def test_coverage_status_on_a_missing_directory_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        rb.coverage_status(_cfg(tmp_path), str(tmp_path / "nope"))
