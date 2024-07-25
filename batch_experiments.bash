#!/bin/bash

python=/isaac-sim/python.sh
# motions=("CShape" "DoubleBendedLine" "PShape" "Angle" "Sine")
motions=("CShape")

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# Iterate over different motions shapes
# for shape in "${motions[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     # $python train.py --model-type discrete   --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.0&
#     # $python train.py --model-type continuous --device cuda:0  --dim-x 8  --total-epochs 10000 --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.0 &
#     # $python train.py --model-type discrete   --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.2&
#     # $python train.py --model-type continuous --device cuda:0  --dim-x 8  --total-epochs 10000 --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.2 &
#     # $python train.py --model-type discrete   --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.5&
#     # $python train.py --model-type continuous --device cuda:0  --dim-x 8  --total-epochs 1000  --expert lasa --motion-shape $shape --experiment-dir boards/batch --ic-noise-rate 0.5&
# done


# Dimension of x
# for shape in "${motions[@]}"; do
#     $python train.py --model-type discrete --device cuda:0  --dim-x 2 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 4 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 16 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 128 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
# done


# for shape in "${motions[@]}"; do
#     $python train.py --model-type discrete --device cuda:0  --dim-x 2 --total-epochs 5000 --experiment-dir boards/dimx-cont --batch-size 1 --num-expert-trajectories 1 &
#     $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 5000 --experiment-dir boards/dimx-cont --batch-size 1 --num-expert-trajectories 1 &
#     $python train.py --model-type discrete --device cuda:0  --dim-x 128 --total-epochs 5000 --experiment-dir boards/dimx-cont --batch-size 1 --num-expert-trajectories 1 &
# done

# Contraction rate
# for shape in "${motions[@]}"; do
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.0 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.05 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.1 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.2 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.4 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.5 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.6 --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/crate --crate-lb 1.8 --batch-size 64&
# done


# for shape in "${motions[@]}"; do
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 1.0 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 1.2 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 1.4 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 1.8 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 2.4 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 3.5 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 4.6 --num-expert-trajectories 1 --batch-size 1&
#     $python train.py --model-type continuous --device cuda:0  --dim-x 8 --total-epochs 2000 --experiment-dir boards/crate_cont --crate-lb 5.7 --num-expert-trajectories 1 --batch-size 1&
# done


# Multi Model Motions
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_1 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_2 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_3 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_4 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&


# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_1 --experiment-dir boards/multi-models --crate-lb 1.2 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_2 --experiment-dir boards/multi-models --crate-lb 1.2 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_3 --experiment-dir boards/multi-models --crate-lb 1.2 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 30000 --motion-shape Multi_Models_4 --experiment-dir boards/multi-models --crate-lb 1.2 --batch-size 70 --num-expert-trajectories 7&

$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_1 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_2 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_3 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_4 --experiment-dir boards/multi-models --crate-lb 1.0 --batch-size 70 --num-expert-trajectories 7&


$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_1 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_2 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_3 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_4 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&

