"""Wiring checks for the pipeline entry points.

These import the real orchestrator, which pulls in the vision stack, so they are
skipped when cv2 / ultralytics are not installed. Everything they assert is about
calls the entry points make, not about the models.
"""

import inspect

import pytest

pytest.importorskip("cv2", reason="orchestrator needs opencv")
pytest.importorskip("ultralytics", reason="orchestrator needs ultralytics")


def source_of(func):
    return inspect.getsource(func)


def test_full_pipeline_seeds_before_doing_work():
    from scripts.run_full_pipeline import main

    src = source_of(main)
    assert "set_seed(" in src
    # seeding must happen before the first pipeline step runs
    assert src.index("set_seed(") < src.index("state = load_state(")


def test_full_pipeline_exposes_seed_and_headless_flags():
    from scripts.run_full_pipeline import main

    src = source_of(main)
    assert "--seed" in src
    assert "--headless" in src


def test_full_pipeline_shares_one_feature_cache_between_vae_stages():
    from scripts.run_full_pipeline import main

    src = source_of(main)
    assert "open_run_feature_cache" in src
    # both stages must receive the same cache object, not two fresh ones
    assert src.count("feature_cache=feature_cache") == 2


def test_full_pipeline_persists_lstm_history():
    from scripts.run_full_pipeline import step_lstm_vae

    assert "lstm_vae_history.csv" in source_of(step_lstm_vae)


def test_flat_vae_saves_its_plot_instead_of_only_showing_it():
    from scripts.run_full_pipeline import step_flat_vae

    src = source_of(step_flat_vae)
    assert "save_path=" in src
    assert "headless=" in src


@pytest.mark.parametrize(
    "module",
    ["scripts.train_vae_pipeline", "scripts.train_lstm_vae_pipeline"],
)
def test_standalone_pipelines_seed_and_cache(module):
    mod = __import__(module, fromlist=["main"])
    src = source_of(mod.main)
    assert "set_seed(" in src
    assert "open_run_feature_cache" in src


def test_lstm_pipeline_persists_history():
    import scripts.train_lstm_vae_pipeline as mod

    assert "save_history(" in source_of(mod.main)
