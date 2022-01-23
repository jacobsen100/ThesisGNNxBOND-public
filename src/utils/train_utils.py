import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torch_geometric.data.batch import Batch as PyGBatch
from src.utils.evaluation import MAE, MSE, SE, AE, RS, RA
from torch.profiler import ProfilerActivity


def train_epoch(
    model,
    dataloader,
    epoch_num,
    total_epochs,
    optimizer,
    loss,
    writer,
    device,
    mean_train,
    profiler,
    log_batch_interval=1,
    gradient_clipping=5,
    print_tqdm=True,
    classification=False,
):
    """
    :param model torch.model:
        model to predict
    :param dataloader torch.utils.data.DataLoader:
        loads the model
    :param epoch_num int:
        current epoch number in outer training loop
    :param optimizer torch.optim.X:
        Optimizer for param training
    :param loss torch.nn.X:
        Loss function for optimization
    :param writer torch.utils.tensorboard.SummaryWriter:
        Writer object to store results from training
    :param device str:
        Device to run training on (CUDA:0/cpu/...)
    :param profiling Class:
        profiling specifications.
    :param log_batch_interval int:
        How often do we log? 1 means every batch, 2 every second and so on..
    :param gradient_clipping float:
        gradient values higher than this will be set to this
    :param print_tqdm boolean:
        if true loading bar will be printed.
    :returns:
        epoch_loss, MSE , MAE
    """
    model.train()

    total_loss = 0
    mse = mae = se = ae = rs = ra = rse = rae = 0

    if print_tqdm:
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        loop.set_description(f"Train epoch [{epoch_num}/{total_epochs}]")
    else:
        loop = enumerate(dataloader)

    for i, data in loop:
        optimizer.zero_grad()

        if isinstance(data, PyGBatch):
            data = data.to(device)
            y = data.y
            out = model(data)

            if len(out) == 2:
                if type(out[0]) == tuple:
                    att_weights = out[0]

                    # a = [torch.mean(elem).item() for elem in att_weights]
                    _y = out[1]
                    backcast_loss = None

                else:

                    backcast_loss = out[0]
                    _y = out[1]

            else:
                backcast_loss = None
                _y = out

        else:
            x = data[0].to(device)
            y = data[1].to(device)

            out = model(x)

            if len(out) == 2:
                if type(out[0]) == tuple:
                    att_weights = out[0]
                    # a = [str(torch.mean(elem).item()) for elem in att_weights]
                    _y = out[1]
                    backcast_loss = None

                else:
                    backcast_loss = out[0]
                    _y = out[1]

            else:
                backcast_loss = None
                _y = out

        if classification:
            batch_loss = loss(_y, y)
        else:
            batch_loss = loss(y, _y)

        if backcast_loss is not None:
            batch_loss = batch_loss + backcast_loss

        batch_loss.backward()

        if gradient_clipping != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()

        if profiler != None:
            profiler.step()

        total_loss += float(batch_loss)

        if i % log_batch_interval == 0:
            # log loss data
            pass

            # TODO Save model?

        if not classification:
            mse += float(MSE(y, _y))
            mae += float(MAE(y, _y))

            # Relative Error
            se += float(SE(y, _y))
            ae += float(AE(y, _y))
            rs += float(RS(y, mean_train))
            ra += float(RA(y, mean_train))

    total_loss /= i + 1
    mse /= i + 1
    mae /= i + 1

    if not classification:
        rse = se / rs
        rae = ae / ra

    if writer:
        writer.add_scalar("Loss/Train", total_loss, epoch_num)
        writer.add_scalar("MSE/Train", mse, epoch_num)
        writer.add_scalar("MAE/Train", mae, epoch_num)

        ########## Fjernet for nu, fordi disse histogrammer fylder pisse meget i tf filen ###########
        # for name, weight in model.named_parameters():
        #    try:
        #        writer.add_histogram(name, weight, epoch_num)
        #    except:
        #        print(f"------ Parameter {name} is NONE. Total loss is : {total_loss}, last batch loss is: {batch_loss} ")
        # if weight.grad is  not None:
        #    writer.add_histogram(f"{name}.grad", weight.grad, epoch_num)

    return total_loss, mse, mae, rse, rae


def validate_epoch(
    model,
    dataloader,
    epoch_num,
    total_epochs,
    loss,
    writer,
    device,
    mean_valid,
    log_batch_interval=1,
    print_tqdm=True,
    classification=False,
):

    """
    :param model torch.model:
        model to predict
    :param dataloader torch.utils.data.DataLoader:
        loads the model
    :param epoch_num int:
        current epoch number in outer training loop
    :param loss torch.nn.X:
        Loss function for optimization
    :param writer torch.utils.tensorboard.SummaryWriter:
        Writer object to store results from training
    :param device str:
        Device to run training on (CUDA:0/cpu/...)
    :param log_batch_interval int:
        How often do we log? 1 means every batch, 2 every second and so on..
    :returns:
        epoch_loss, MSE , MAE
    """

    model.eval()
    total_loss = 0
    mse = mae = se = ae = rs = ra = rse = rae = 0

    if print_tqdm:
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        loop.set_description(f"Valid epoch [{epoch_num}/{total_epochs}]")
    else:
        loop = enumerate(dataloader)

    with torch.no_grad():
        for i, data in loop:

            if isinstance(data, PyGBatch):
                data = data.to(device)
                y = data.y
                out = model(data)

                if len(out) == 2:
                    if type(out[0]) == tuple:
                        att_weights = out[0]
                        _y = out[1]
                        backcast_loss = None
                    else:
                        backcast_loss = out[0]
                        _y = out[1]
                else:
                    backcast_loss = None
                    _y = out

            else:
                x = data[0].to(device)
                y = data[1].to(device)
                out = model(x)

                if len(out) == 2:
                    if type(out[0]) == tuple:
                        att_weights = out[0]
                        _y = out[1]
                        backcast_loss = None
                    else:
                        backcast_loss = out[0]
                        _y = out[1]
                else:
                    backcast_loss = None
                    _y = out

            if classification:
                batch_loss = loss(_y, y)
            else:
                batch_loss = loss(_y, y)

            if backcast_loss is not None:
                batch_loss = batch_loss + backcast_loss

            total_loss += float(batch_loss)

            if i % log_batch_interval == 0:
                # log loss batch data
                pass

            # TODO Save model?
            if not classification:
                mse += float(MSE(y, _y))
                mae += float(MAE(y, _y))

                # Relative Error
                se += float(SE(y, _y))
                ae += float(AE(y, _y))
                rs += float(RS(y, mean_valid))
                ra += float(RA(y, mean_valid))

    total_loss /= i + 1
    mse /= i + 1
    mae /= i + 1
    if not classification:
        rse = se / rs
        rae = ae / ra

    if writer:
        writer.add_scalar("Loss/Validation", total_loss, epoch_num)
        writer.add_scalar("MSE/Validation", mse, epoch_num)
        writer.add_scalar("MAE/Validation", mae, epoch_num)
    return total_loss, mse, mae, rse, rae
