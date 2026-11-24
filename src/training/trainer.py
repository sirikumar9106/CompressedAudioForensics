"""Generic training loop: dev-EER checkpoint selection + early stopping. (Phase 3/4 — not run yet.)"""
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.metrics import eer


@torch.no_grad()
def score_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    S, Y = [], []
    for y, lab in loader:
        lg = model(y.to(device))
        S.append((lg[:, 1] - lg[:, 0]).cpu().numpy()); Y.append(lab.numpy())
    return np.concatenate(Y), np.concatenate(S)


def train(model, train_loader, dev_loader, criterion, optimizer, device, out_dir,
          epochs: int = 20, patience: int = 5, log_every: int = 100):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    best, bad, hist = float("inf"), 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for i, (y, lab) in enumerate(train_loader, 1):
            loss = criterion(model(y.to(device)), lab.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item()
            if i % log_every == 0:
                print(f"ep{ep} it{i} loss {tot / i:.4f}")
        labels, scores = score_loader(model, dev_loader, device)
        dev_eer = eer(labels, scores)[0]
        hist.append({"epoch": ep, "train_loss": tot / max(len(train_loader), 1), "dev_eer": dev_eer})
        print(hist[-1])
        if dev_eer < best:
            best, bad = dev_eer, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    return best
