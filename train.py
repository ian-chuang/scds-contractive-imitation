#!/usr/bin/env python
import os
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tslearn.metrics import SoftDTWLossPyTorch

from ren_discrete import DREN
from ren_continuous import CREN

from ren_trainer import train_ren_model
from cli import argument_parser
from dataset import lasa_expert


# main entry
if __name__ == '__main__':

    # parse and set experiment arguments
    args = argument_parser()

    # set expert traj
    if args.expert == "lasa":
        dataloader = lasa_expert(motion_shape=args.motion_shape, horizon=args.horizon,
                                 device=args.device, batch_size=args.batch_size,
                                 num_exp_trajectories=args.num_expert_trajectories,
                                 num_aug_trajectories=args.num_augment_trajectories,
                                 ic_noise_rate=args.ic_noise_rate)

        # sanity check for the dataset
        ic, traj = next(iter(dataloader))
        print(f'Expert data size [{ic.shape}, {traj.shape}], total of {len(dataloader.dataset)} entries')

    else:
        raise(NotImplementedError(f'Expert data is not available!'))

    # define REN model
    if args.model_type == 'continuous':
        model = CREN(dim_in=args.dim_in, dim_out=args.dim_out, dim_x=args.dim_x, dim_v=args.dim_v,
                     batch_size=args.batch_size, device=args.device, horizon=args.horizon,
                     contraction_rate_lb=args.crate_lb, bijection=args.bijection,
                     num_bijection_layers=args.num_bijection_layers)

    elif args.model_type == 'discrete':
        model = DREN(dim_in=args.dim_in, dim_out=args.dim_out, dim_x=args.dim_x, dim_v=args.dim_v,
                     batch_size=args.batch_size, device=args.device, horizon=args.horizon,
                     contraction_rate_lb=args.crate_lb, bijection=args.bijection,
                     num_bijection_layers=args.num_bijection_layers)

    else:
        raise(NotImplementedError('Please determine a correct model type: ["continuous", "discrete"]!'))

    # send the model to device
    model.to(device=args.device)

    # experiment log setup
    timestamp = datetime.now().strftime('%d-%H%M')
    experiment_name = f'{type(model).__name__.lower()}-{args.expert}-{args.motion_shape}' \
                      f'-h{args.horizon}-x{args.dim_x}-e{args.total_epochs}-b{args.batch_size}' \
                      f'-cr{args.crate_lb}-e{args.num_expert_trajectories}-s{args.num_expert_samples}' \
                      f'-a{args.num_augment_trajectories}-t{timestamp}'

    writer_dir = f'{args.experiment_dir}/{experiment_name}'
    writer = SummaryWriter(writer_dir)

    # loss function
    loss = torch.nn.MSELoss() if args.loss == "mse" else SoftDTWLossPyTorch(normalize=True, gamma=0.1)

    # training loop
    ren_trained, ren_data = train_ren_model(model=model, lr=args.lr, horizon=args.horizon,
                                            expert_data=dataloader, total_epochs=args.total_epochs,
                                            lr_start_factor=args.lr_start_factor, writer=writer,
                                            lr_end_factor=args.lr_end_factor,
                                            patience_epoch=args.patience_epoch,
                                            log_epoch=args.log_epoch,
                                            criterion=loss)

    ren_data["expert"] = args.expert
    ren_data["num_expert_trajectories"] = args.num_expert_trajectories
    ren_data["num_augment_trajectories"] = args.num_augment_trajectories
    ren_data["ic_noise_rate"] = args.ic_noise_rate

    if args.expert == "lasa":
        ren_data["motion_shape"] = args.motion_shape
    else:
        raise NotImplementedError(f'Expert is not fully implemented at this stage!')

    # save the data
    torch.save(ren_data, os.path.join(writer_dir, 'best_model.pth'))
