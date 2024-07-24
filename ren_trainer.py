import copy
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from typing import Union

from ren_discrete import DREN
from ren_continuous import CREN


def train_ren_model(model: Union[DREN, CREN], lr: float, u_in: torch.Tensor,
                    y_init: torch.Tensor, horizon: int, expert_trajectory: torch.Tensor,
                    total_epochs: int, lr_start_factor: float, lr_end_factor: float,
                    patience_epoch: int, log_epoch: int, ic_noise_rate: float, writer: SummaryWriter,
                    device: str, batch_size: int):
    """ Train a discrete or continuous ren model.

    # TODO: Train a scaling network at the same time to remember the magnitude of velocity.

    Args:
        model (CREN, DREN): The trainable model.
        lr (float): Learning rate for the optimizer.
        u_in (torch.Tensor): Exogenous input of the ren.
        y_init (torch.Tensor): Initial condition in the output space.
        horizon (int): Horizon or the length of each trajectory.
        expert_trajectory (torch.Tensor): Expert trajectories for training.
        total_epochs (int): Total number of epochs.
        lr_start_factor (float): Learning rate start factor.
        lr_end_factor (float): Learning rate end factor.
        patience_epoch (int): Tolerance toward no progress.
        log_epoch (int): Log rate in number of epochs.
        noise_ratio (float): Ratio of the noise on the output or internal state.
    """

    print(f'Training { type(model).__name__} model for {total_epochs} epochs and {horizon} samples')

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

    # repeat expert data according to the batch size
    stacked_expert_trajectory = expert_trajectory.repeat(batch_size // expert_trajectory.shape[0], 1, 1)
    stacked_y_init = y_init.repeat(batch_size // y_init.shape[0], 1, 1)

    # patience and log epochs
    patience_epoch = (total_epochs // 5) if patience_epoch is None else patience_epoch
    log_epoch = (total_epochs // 10) if log_epoch is None else log_epoch

    # training epochs
    for epoch in range(total_epochs):
        # zero grad
        optimizer.zero_grad()

        # forward pass
        # TODO: Then generate these random initial conditions as a part of dataloader
        y_init_noisy = stacked_y_init + ic_noise_rate * (2 * (torch.rand(batch_size, y_init.shape[1], y_init.shape[2], device=device) - 0.5))
        out = model.forward_trajectory(u_in, y_init_noisy, horizon)

        # loss
        loss = criterion(out, stacked_expert_trajectory)

        train_losses.append(loss.item())

        # best model
        if best_loss - loss > 5e-6:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_loss = loss
            best_train_epoch = epoch

        # check no progress
        if epoch - best_train_epoch > patience_epoch:
            print(f'No significant progress in a while, aborting training')
            break

        # backward and steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_model_param()

        # logs
        if epoch % log_epoch == 0:
            print(f'Epoch: {epoch}/{total_epochs} | Best Loss: {best_loss:.8f} | Best Epoch: {best_train_epoch} | LR: {scheduler.get_last_lr()[0]:.6f}')
            trajectories.append(out.detach().cpu().numpy())

        # tensorboard
        writer.add_scalars('Training Loss', {'Training': loss.item(), 'Best': best_loss.item()}, epoch + 1)
        writer.flush()

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
        'ic_noise_rate': ic_noise_rate
    }

    # load the best model for plotting
    model.load_state_dict(best_model_state_dict)
    model.update_model_param()

    return model, model_data
