import copy
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Union

from ren_discrete import REN
from node_ren import NODE_REN

from torchdiffeq import odeint_adjoint as odeint


def train_ren_model(ren_module: Union[REN, NODE_REN], ren_lr: float, u_in: torch.Tensor,
                    y_init: torch.Tensor, ren_horizon: int, expert_trajectory: torch.Tensor,
                    total_epochs: int, ren_lr_start_factor: float, ren_lr_end_factor: float,
                    patience_epoch: int, log_epoch: int, ic_noise_rate: float, writer: SummaryWriter,
                    device: str, batch_size: int):
    """ Train a discrete ren model.

    Args:
        ren_module (REN): The trainable model.
        ren_lr (float): Learning rate.
        u_in (torch.Tensor): Exogenous input of the ren.
        y_init (torch.Tensor): Initial condition in the output space.
        ren_horizon (int): Horizon or the length of each trajectory.
        expert_trajectory (torch.Tensor): Expert trajectories for training.
        total_epochs (int): Total number of epochs.
        ren_lr_start_factor (float): Learning rate start factor.
        ren_lr_end_factor (float): Learning rate end factor.
        patience_epoch (int): Tolerance toward no progress.
        log_epoch (int): Log rate in number of epochs.
        noise_ratio (float): Ratio of the noise on the output or internal state.
    """

    # optimizer
    optimizer = torch.optim.Adam(ren_module.parameters(), lr=ren_lr)

    # loss
    criterion = nn.MSELoss()

    # temps
    trajectories, train_losses = [], []
    best_model_state_dict = None
    best_loss = torch.tensor(float('inf'))
    best_train_epoch = 0

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=ren_lr_start_factor,
                                                  end_factor=ren_lr_end_factor, total_iters=total_epochs)

    # time the operation
    start_time = datetime.now()

    stacked_expert_trajectory = expert_trajectory.repeat(batch_size, 1, 1)

    # training epochs
    for epoch in range(total_epochs):
        # zero grad
        optimizer.zero_grad()

        # forward pass
        # TODO: Then generate these random initial conditions as a part of dataloader
        y_init_noisy = y_init + ic_noise_rate * (2 * (torch.rand(batch_size, y_init.shape[1], y_init.shape[2], device=device) - 0.5))
        out = ren_module.forward_trajectory(u_in, y_init_noisy, ren_horizon)

        # loss
        loss = criterion(out, stacked_expert_trajectory)

        train_losses.append(loss.item())

        # best model
        if loss < best_loss:
            best_model_state_dict = copy.deepcopy(ren_module.state_dict())
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
        ren_module.update_model_param()

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
    best_state = {
        'model_state_dict': best_model_state_dict,
        'train_trajectories': trajectories,
        'train_losses': train_losses,
        'best_loss': best_loss,
        'training_time': training_time.total_seconds(),
        'training_epochs': epoch + 1
    }

    file = f'{writer.get_logdir()}/best_model.pth'
    torch.save(best_state, file)

    # load the best model for plotting
    ren_module.load_state_dict(best_model_state_dict)
    ren_module.update_model_param()

    return ren_module