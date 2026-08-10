"""MineRL trajectory builders.

The MineRL package owns acquisition and decoding of its demonstration archive.
This module deliberately imports it lazily, then turns each transition into a
cacheable StableDataset row.  The resulting rows retain episode and timestep
metadata, so downstream users can form temporal windows without guessing at
trajectory boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from stable_datasets.schema import DatasetInfo, DatasetSource, Features, Image, Sequence, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


class MineRLDataPipeline(Protocol):
    """The small MineRL data API required by the dataset builder."""

    def get_trajectory_names(self) -> list[str]: ...

    def load_data(self, stream_name: str) -> Iterable[tuple[Any, ...]]: ...


def _flatten_numeric(value: Any) -> np.ndarray:
    """Flatten nested numeric MineRL actions in a stable key order."""
    if isinstance(value, Mapping):
        parts = [_flatten_numeric(value[key]) for key in sorted(value)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    if isinstance(value, tuple | list):
        parts = [_flatten_numeric(item) for item in value]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"MineRL action values must be numeric, got {array.dtype!s}.")
    return array.astype(np.float32, copy=False).reshape(-1)


def _rgb_pov(state: Mapping[str, Any]) -> np.ndarray:
    """Return MineRL's ``pov`` observation as HWC uint8 RGB."""
    if "pov" not in state:
        raise KeyError("MineRL state has no 'pov' observation.")
    frame = np.asarray(state["pov"])
    if frame.ndim != 3:
        raise ValueError(f"MineRL state['pov'] must be rank-3, got {frame.shape}.")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] != 3:
        raise ValueError(f"MineRL state['pov'] must have three channels, got {frame.shape}.")
    return frame.astype(np.uint8, copy=False)


def _make_pipeline(environment: str, data_dir: Path | None) -> MineRLDataPipeline:
    try:
        import minerl
    except ImportError as exc:
        raise ImportError(
            "MineRLTreechop requires MineRL. Install MineRL and download "
            "MineRLTreechop-v0 before constructing this dataset."
        ) from exc
    return minerl.data.make(environment, data_dir=str(data_dir) if data_dir else None, num_workers=1)


class MineRLTreechop(BaseDatasetBuilder):
    """MineRLTreechop-v0 human demonstrations as transition-level rows.

    MineRL is not redistributed. ``data_dir`` must point to a locally acquired
    MineRL data root (or MineRL's configured default data directory is used).
    Every row is one ``(observation_t, action_t, reward_t, done_t)`` record;
    ``episode_id`` and ``timestep`` make temporal reconstruction explicit.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://minerl.readthedocs.io/",
        assets={},
        license="MineRL data terms apply. This builder does not redistribute demonstrations.",
        citation="""@inproceedings{guss2019minerl,
  title={MineRL: A Large-Scale Dataset of Minecraft Demonstrations},
  author={Guss, William H. and others},
  booktitle={IJCAI},
  year={2019}
}""",
    )

    def __init__(self, config_name: str | None = None, data_dir: str | Path | None = None, **kwargs):
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        super().__init__(config_name=config_name, **kwargs)

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=(
                "MineRLTreechop-v0 transitions with first-person RGB observations and "
                "deterministically flattened MineRL actions."
            ),
            features=Features(
                {
                    "image": Image(),
                    "action": Sequence(Value("float32")),
                    "reward": Value("float32"),
                    "done": Value("bool"),
                    "episode_id": Value("int32"),
                    "trajectory_name": Value("string"),
                    "timestep": Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN]

    def _split_generators(self) -> list[SplitGenerator]:
        pipeline = _make_pipeline("MineRLTreechop-v0", self.data_dir)
        names = sorted(pipeline.get_trajectory_names())
        if not names:
            raise ValueError("No MineRLTreechop-v0 trajectories were found.")
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"pipeline": pipeline, "trajectory_names": names})]

    def _generate_examples(
        self,
        pipeline: MineRLDataPipeline,
        trajectory_names: list[str],
    ):
        action_dim: int | None = None
        for episode_id, trajectory_name in enumerate(trajectory_names):
            for timestep, transition in enumerate(pipeline.load_data(trajectory_name)):
                if len(transition) < 5:
                    raise ValueError(
                        "MineRL transitions must be (state, action, reward, next_state, done)."
                    )
                state, action, reward, _next_state, done = transition[:5]
                if not isinstance(state, Mapping):
                    raise TypeError("MineRL state must be a mapping containing 'pov'.")
                action_vector = _flatten_numeric(action)
                if action_dim is None:
                    action_dim = int(action_vector.size)
                elif action_vector.size != action_dim:
                    raise ValueError(
                        "MineRL action schema changed within the dataset: "
                        f"expected {action_dim} values, got {action_vector.size}."
                    )
                key = f"{episode_id}:{timestep}"
                yield key, {
                    "image": _rgb_pov(state),
                    "action": action_vector,
                    "reward": np.float32(reward),
                    "done": bool(done),
                    "episode_id": np.int32(episode_id),
                    "trajectory_name": trajectory_name,
                    "timestep": np.int32(timestep),
                }


__all__ = ["MineRLTreechop"]
