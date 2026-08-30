"""exp001_smoke — proves the whole loop works end to end.

Run it:  task run -- exp001_smoke   (or: uv run python experiments/exp001_smoke/run.py)
"""

from __future__ import annotations

from pathlib import Path

import torch
from sconixlib import Run, get_device, load_config, set_seed
from torch.utils.data import DataLoader, TensorDataset

from vllm_explore.data import make_regression
from vllm_explore.model import MLP

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", HERE / "config.yaml")
    set_seed(cfg["seed"])
    device = get_device()

    X, y = make_regression(cfg["dataset"]["n"], cfg["dataset"]["dim"], cfg["dataset"]["noise"])
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = MLP(cfg["dataset"]["dim"], cfg["hidden"], cfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.MSELoss()

    with Run("exp001_smoke", config=cfg, tags=["smoke"]) as run:
        run.summary(device=str(device), params=sum(p.numel() for p in model.parameters()))
        for epoch in range(cfg["epochs"]):
            model.train()
            total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            epoch_loss = total / len(ds)
            run.log(step=epoch, loss=epoch_loss)
            print(f"epoch {epoch:3d}  loss {epoch_loss:.5f}")
        run.summary(final_loss=epoch_loss)


if __name__ == "__main__":
    main()
