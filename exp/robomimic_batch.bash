#!/bin/bash

python=/isaac-sim/python.sh
robomimic_tasks=("can" "square") # "lift" "square" "transport")

# Switch to main dir
cd .. || { echo "Failed to change directory"; exit 1; }
echo "Current working directory: $(pwd)"

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# EEF_POS Data
for shape in "${robomimic_tasks[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    python train.py --expert robomimic --motion-shape $shape --model-type continuous  --dim-in 3 --dim-out 3 --device cuda:0  --dim-x 128  --total-epochs 2000  --experiment-dir results/robomimic_motions/simreal --bijection --num-bijection-layers 8 --crate-lb 12.0  --num-expert-trajectories 1 --horizon 20 --loss dtw &
    python train.py --expert robomimic --motion-shape $shape --model-type continuous  --dim-in 3 --dim-out 3 --device cuda:0  --dim-x 128  --total-epochs 2000  --experiment-dir results/robomimic_motions/simreal --bijection --num-bijection-layers 8 --crate-lb 15.0  --num-expert-trajectories 1 --horizon 20 --loss dtw &

    # python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dim-in 3 --dim-out 3 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/single_expert --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 5 --horizon 50 --loss dtw&
    # python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dim-in 3 --dim-out 3 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/single_expert --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 10 --horizon 50 --loss dtw&
    # python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dim-in 3 --dim-out 3 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/single_expert --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 20 --horizon 50 --loss dtw&
done


# # EEF_POSE Data
# for shape in "${robomimic_tasks[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key eef_pos_ori --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/pose_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 1 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key eef_pos_ori --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/pose_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 5 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key eef_pos_ori --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/pose_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 10 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key eef_pos_ori --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/pose_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 20 --horizon 50 --loss dtw&
# done

# # JOINT_POS Data
# for shape in "${robomimic_tasks[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key joint_pos --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/joint_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 1 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key joint_pos --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/joint_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 5 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key joint_pos --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/joint_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 10 --horizon 50 --loss dtw&
#     python train.py --expert robomimic --motion-shape $shape --model-type discrete  --dataset-key joint_pos --dim-in 6 --dim-out 6 --device cuda:0  --dim-x 64  --total-epochs 15000  --experiment-dir results/robomimic_motions/joint_planning --bijection --num-bijection-layers 8 --crate-lb 1.0  --num-expert-trajectories 20 --horizon 50 --loss dtw&
# done