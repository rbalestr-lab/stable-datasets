"""Tests for MineRL transition caching without a MineRL installation."""

from __future__ import annotations

import numpy as np
import pytest

from stable_datasets.video.minerl import MineRLTreechop, _flatten_numeric


def _transition(value: int, *, done: bool = False):
    state = {"pov": np.full((4, 5, 3), value, dtype=np.uint8)}
    action = {
        "camera": np.array([value, -value], dtype=np.float32),
        "forward": value % 2,
        "jump": bool(value % 2),
    }
    return state, action, float(value), state, done


class _Pipeline:
    def get_trajectory_names(self):
        return ["second", "first"]

    def load_data(self, name):
        return [_transition(1), _transition(2, done=name == "second")]


def test_minerl_treechop_caches_transition_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_datasets.video.minerl._make_pipeline", lambda *args: _Pipeline())

    ds = MineRLTreechop(split="train", processed_cache_dir=tmp_path / "processed")

    assert len(ds) == 4
    sample = ds.with_format("numpy")[0]
    assert set(sample) == {
        "image",
        "action",
        "reward",
        "done",
        "episode_id",
        "trajectory_name",
        "timestep",
    }
    assert sample["trajectory_name"] == "first"
    assert sample["image"].shape == (4, 5, 3)
    np.testing.assert_allclose(sample["action"], [1, -1, 1, 1])
    assert sample["episode_id"] == 0
    assert sample["timestep"] == 0


def test_minerl_actions_reject_non_numeric_values():
    with pytest.raises(TypeError, match="numeric"):
        _flatten_numeric({"forward": "yes"})
