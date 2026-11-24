"""Dataset returning (y, label). y = cached platform-degraded audio; y' is generated in-graph by DeltaNet
so training and inference share the exact same perturbation code."""
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.benchmark_guard import assert_not_locked
from .audio_processing import load_audio, preprocess
from .offline_cache import cache_path


class DegradedAudioDataset(Dataset):
    def __init__(self, manifest: str, codec: str | None = "mp3", bitrate: str = "32k",
                 cache_root: str = "data/cache", train: bool = True, seed: int = 0, rows: list | None = None,
                 clean_prob: float = 0.0):
        assert_not_locked(manifest)
        self.clean_prob = clean_prob  # Baseline C: fraction of samples drawn from clean audio
        self.rows = rows if rows is not None else list(csv.DictReader(open(manifest, newline="")))
        self.codec, self.bitrate, self.cache_root, self.train = codec, bitrate, cache_root, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def path_for(self, r) -> str:
        if self.codec is None:  # clean condition
            return r["path"]
        return str(cache_path(self.cache_root, self.codec, self.bitrate, r["file_id"]))

    def __getitem__(self, i):
        r = self.rows[i]
        clean = self.train and self.clean_prob > 0 and self.rng.random() < self.clean_prob
        x = preprocess(load_audio(r["path"] if clean else self.path_for(r)), mode="random" if self.train else "center", rng=self.rng)
        return torch.from_numpy(x), int(r["label"])

    def class_weights(self) -> torch.Tensor:
        """w_c = N / (2 * n_c); index 0 = spoof, 1 = bonafide (~0.56 / ~4.92 on 2019 LA train)."""
        y = np.array([int(r["label"]) for r in self.rows])
        n = len(y)
        return torch.tensor([n / (2 * max((y == 0).sum(), 1)), n / (2 * max((y == 1).sum(), 1))], dtype=torch.float32)
