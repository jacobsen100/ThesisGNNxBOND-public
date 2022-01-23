import hydra
import os
import math
import time
import warnings
import pickle
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr

from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

import logging

log = logging.getLogger(__name__)

from src.data import TimeSeriesDataset, Transformation
from src.utils import (
    model_builder,
    optimizer_builder,
    transforms_builder,
    train_epoch,
    validate_epoch,
    dataloader_builder,
)

# ignore geometric laoder warning:
warnings.filterwarnings("ignore", ".* Please explicitly set 'num_nodes'.*")


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    OG_path = hydra.utils.get_original_cwd()
    filename = ":".join(os.getcwd().split("/")[-2:])
    loc = os.path.join(OG_path, "runs", filename)

    print("\n" + "#" * 62)
    print("############ Training with current configuration: ################")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 62)
    print(f"Experiment : {filename}")
    print(
        f"Processed files in : w:{cfg.data.window_size}-h:{cfg.data.horizon}-cw:{cfg.data.window_size}-ct:{int(cfg.data.correlation_threshold*100)}"
    )
    print("#" * 62, "\n")

    ########################################### VARS ###########################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"################ Device : {device} ############")
    print_tqdm = False
    model_output = cfg.training.output_model
    save_model_after_epoch = cfg.training.save_model_after_epoch  # 0 indexed
    max_epochs_no_improvement = cfg.training.prune_after_epochs

    writer = SummaryWriter(loc)
    # writer = None

    ########################################### DATA ###########################################
    """ These are defined in data_builder instead now 
    window_size = cfg.data.window_size
    horizon = cfg.data.horizon
    shuffle = cfg.data.shuffle_loader
    batch_size = cfg.training.batch_size
    """
    transforms_list = transforms_builder(cfg)
    transformer = Transformation(transforms_list, cfg, device)

    trainloader, _ = dataloader_builder(cfg, transformer, OG_path, "train")

    # Save transformer after fitting on traing data to be able to backtransform
    with open("transformer.pkl", "wb") as handle:
        pickle.dump(transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    validloader, _ = dataloader_builder(cfg, transformer, OG_path, "valid")

    ########################################## MODEL + OPTIM ##########################################
    model = model_builder(cfg).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    log.info(
        f"####### NUMBER OF PARAMETERS {sum([p.numel() for p in model.parameters() if p.requires_grad])} #######"
    )

    optimizer = optimizer_builder(cfg, parameters)

    if cfg.training.scheduler_gamma == 1:
        lr_scheduler = None
    elif cfg.training.scheduler_gamma < 1:
        log.info(
            "++++++++++++++++++++++++++++++++++ Using Exponential lr_scheduler ++++++++++++++++++++++++++++++++++"
        )
        lr_scheduler = lr.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)
    elif cfg.training.scheduler_gamma == 2:
        log.info(
            "++++++++++++++++++++++++++++++++++ Using Multistep lr_scheduler ++++++++++++++++++++++++++++++++++"
        )
        lr_scheduler = lr.MultiStepLR(
            optimizer, milestones=[30, 60, 90, 120], gamma=0.2
        )
    elif cfg.training.scheduler_gamma == 3:
        log.info(
            "++++++++++++++++++++++++++++++++++ Using Cyclical lr_scheduler ++++++++++++++++++++++++++++++++++"
        )
        lr_scheduler = lr.CyclicLR(
            optimizer,
            base_lr=0.00001,
            max_lr=0.001,
            step_size_up=10,
            step_size_down=10,
            mode="exp_range",
            gamma=0.95,
            cycle_momentum=False,
        )

    elif cfg.training.scheduler_gamma == 4:
        log.info(
            "++++++++++++++++++++++++++++++++++ Using Plateau lr_scheduler ++++++++++++++++++++++++++++++++++"
        )
        lr_scheduler = lr.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=25,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            verbose=True,
        )
    if cfg.data.classification:
        loss_func = nn.BCELoss()
    else:
        loss_func = nn.MSELoss()
    ########################################### Training ###########################################
    epochs = cfg.training.epochs
    best_val = math.inf

    is_profiler = cfg.training.profiling
    if is_profiler:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1,  # during this phase profiler is not active
                warmup=1,  # during this phase profiler starts tracing, but the results are discarded
                active=3,  # during this phase profiler traces and records data
                repeat=1,
            ),  # specifies an upper bound on the number of cycles
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling_log/"),
            with_stack=True,  # enable stack tracing, adds extra profiling overhead
        )
    else:
        profiler = None
    if is_profiler:
        profiler.start()

    num_epochs_with_no_improvement = 0
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        ######### TRAIN
        train_start = time.time()
        train_total_loss, train_MSE, train_MAE, train_RSE, train_RAE = train_epoch(
            model=model,
            dataloader=trainloader,
            epoch_num=epoch,
            total_epochs=epochs,
            optimizer=optimizer,
            loss=loss_func,
            writer=writer,
            device=device,
            log_batch_interval=1,
            gradient_clipping=cfg.training.gradient_clipping,
            print_tqdm=print_tqdm,
            profiler=profiler,
            mean_train=transformer.mean_train,
            classification=cfg.data.classification,
        )

        log.info(
            f"| ### Train epoch : {epoch:4.0f}  |  Time : {(time.time()-train_start):>8.4f}  |  MSE : {train_total_loss:8.6f}  |  MAE : {train_MAE:8.6f}  |  RSE : {train_RSE:8.6f}  |  RAE : {train_RAE:8.6f} ### |"
        )

        ####### VALIDATE
        valid_start = time.time()

        valid_total_loss, valid_MSE, valid_MAE, valid_RSE, valid_RAE = validate_epoch(
            model=model,
            dataloader=validloader,
            epoch_num=epoch,
            total_epochs=epochs,
            loss=loss_func,
            writer=writer,
            device=device,
            log_batch_interval=1,
            print_tqdm=print_tqdm,
            mean_valid=transformer.mean_valid,
            classification=cfg.data.classification,
        )

        log.info(
            f"| --- Valid epoch : {epoch:4.0f}  |  Time : {(time.time()-valid_start):>8.4f}  |  MSE : {valid_total_loss:8.6f}  |  MAE : {valid_MAE:8.6f}  |  RSE : {valid_RSE:8.6f}  |  RAE : {valid_RAE:8.6f} --- |"
        )

        log.info("-" * 108)

        if (
            valid_total_loss < best_val
        ):  # only save after `save_model_after_epoch` epochs
            best_val = valid_total_loss
            num_epochs_with_no_improvement = 0
            if epoch > save_model_after_epoch:
                if model_output:
                    torch.save(model.state_dict(), "model.pt")
                    log.info(
                        "- - - - - - - - - - - - - - - - - - - - - - MODEL SAVED - - - - - - - - - - - - - - - - - - - - - -"
                    )
                else:
                    log.info(
                        "- - - - - - - - - - - - - - - - - - - - - - BEST SCORE - - - - - - - - - - - - - - - - - - - - - -"
                    )

        else:
            num_epochs_with_no_improvement += 1

        if num_epochs_with_no_improvement > max_epochs_no_improvement:
            log.info(
                "/ / / / / / / / / / / / / / / / / / / / / / NO IMPROVEMENT - STOPPING RUN / / / / / / / / / / / / / / / / / / / / / /"
            )
            break

        if lr_scheduler is not None:
            if cfg.training.scheduler_gamma == 4:
                lr_scheduler.step(valid_total_loss)
            else:
                lr_scheduler.step()

    if is_profiler:
        profiler.stop()

    log.info(f"Best validation loss: {best_val}")

    if model_output:
        torch.save(model.state_dict(), "model_last.pt")

    if writer:
        writer.close()

    return best_val


if __name__ == "__main__":
    main()
