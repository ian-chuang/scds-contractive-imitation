import copy
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from typing import Union

from torch.utils.data import DataLoader

from ren_discrete import DREN
from ren_continuous import CREN


def train_ren_model(model: Union[DREN, CREN], lr: float, horizon: int,
                    expert_data: DataLoader, total_epochs: int,
                    lr_start_factor: float, lr_end_factor: float,
                    patience_epoch: int, log_epoch: int,
                    writer: SummaryWriter):
    """ Train a discrete or continuous ren model.

    Args:
        model (CREN, DREN): The trainable model.
        u_in (torch.Tensor): Exogenous input of the ren.
        horizon (int): Horizon or the length of each predicted trajectory.
        expert_data (DataLoader): Expert trajectories for training.

        lr (float): Learning rate for the optimizer.
        lr_start_factor (float): Learning rate start factor.
        lr_end_factor (float): Learning rate end factor.

        total_epochs (int): Total number of epochs.
        patience_epoch (int): Tolerance toward no progress.
        log_epoch (int): Log rate in number of epochs.

        device (str): Name of the computation device.
    """

    print(f'Training { type(model).__name__} model for {total_epochs} epochs and {horizon} samples on {device}')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss
    criterion = nn.MSELoss()

    # temps
    trajectories, train_losses = [], []
    best_model_state_dict = None
    best_loss = torch.tensor(float('inf'))
    best_train_epoch = 0

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_start_factor,
                                                  end_factor=lr_end_factor, total_iters=total_epochs)

    # time the operation
    start_time = datetime.now()

    # patience and log epochs
    patience_epoch = (total_epochs // 10) if patience_epoch is None else patience_epoch
    log_epoch = (total_epochs // 10) if log_epoch is None else log_epoch

    # training epochs
    step = 0
    for epoch in range(total_epochs):
        for y_init, expert_trajectory in expert_data:
            # zero grad
            optimizer.zero_grad()

            # input is set to zero
            u_in = torch.zeros((y_init.size(0), 1, model.dim_in), device=model.device)

            # forward pass
            out = model.forward_trajectory(u_in, y_init, horizon)

            # loss
            loss = criterion(out, expert_trajectory)
            train_losses.append(loss.item())

            # best model
            if best_loss - loss > 5e-6:
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_loss = loss.item()
                best_train_epoch = epoch
                best_out = out.detach().cpu()

            # tensorboard
            writer.add_scalars('Loss', {'Training': best_loss}, step + 1)
            writer.flush()
            step += 1

            # backward and param updates
            loss.backward()
            optimizer.step()
            model.update_model_param()

        # check no progress
        if epoch - best_train_epoch > patience_epoch:
            print(f'No significant progress in a while, aborting training')
            break

        # logs
        if epoch % log_epoch == 0:
            print(f'Epoch: {epoch}/{total_epochs} | Best Loss: {best_loss:.8f} | Best Epoch: {best_train_epoch} | LR: {scheduler.get_last_lr()[0]:.6f}')
            trajectories.append(best_out)

        # step the lr scheduler
        scheduler.step()

    # training time and best results
    training_time = datetime.now() - start_time
    print(f'Training Concluded in {training_time} | Best Loss: {best_loss:.8f} | Best Epoch: {best_train_epoch}')

    # save the best model
    model_data = {
        'model_name': type(model).__name__,
        'model_state_dict': best_model_state_dict,
        'model_params': model.get_init_params(),
        'train_trajectories': trajectories,
        'train_losses': train_losses,
        'best_loss': best_loss,
        'training_time': training_time.total_seconds(),
        'training_epochs': epoch + 1,
    }

    # load the best model for plotting
    model.load_state_dict(best_model_state_dict)
    model.update_model_param()

    return model, model_data
