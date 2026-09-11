import torch
import glob
import os

ckpts = sorted(glob.glob("/scratch/user/motahare/emma/checkpoints/best_server_*mobileclip*.pt"))
for p in ckpts:
    c = torch.load(p, map_location="cpu", weights_only=False)
    ep = c.get("epoch", "?")
    vm = c.get("val_metrics", {})
    acc = vm.get("accuracy", vm.get("acc", "?"))
    name = os.path.basename(p)
    print(f"{name:60s}  epoch={ep:>3}  val_acc={acc}")
