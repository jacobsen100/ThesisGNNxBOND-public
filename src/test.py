import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="config", config_name="config.yaml")
def get_dataset(cfg: DictConfig):
    optim = cfg.optimizer

    print("Optim:", optim)

    lr = cfg.optimizer.lr
    print("lr ", lr)

    mom = cfg.optimizer.momentum

    print("mom ", mom)
    OG_path = hydra.utils.get_original_cwd()
    filename = ":".join(os.getcwd().split("/")[-2:])
    loc = os.path.join(OG_path, "runs", filename)
    writer = SummaryWriter(loc)
    writer.add_scalar("Salami", 8)


if __name__ == "__main__":
    get_dataset()
