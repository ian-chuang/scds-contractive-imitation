#!/bin/bash

python=/isaac-sim/python.sh
basic_motions=("GShape" "DoubleBendedLine" "PShape" "Angle" "Sine" "Worm" "Snake" "NShape")
mm_motions=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# Basic motion shape experiments
# for shape in "${basic_motions[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/single_models --bijection --num-bijection-layers 4 --crate-lb 1.0  --num-expert-trajectories 4&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/single_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 4&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/single_models --bijection --num-bijection-layers 4 --crate-lb 1.0  --num-expert-trajectories 1&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/single_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 1&
# done

# Multi model shape experiments
# for shape in "${mm_motions[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 4 --crate-lb 1.0  --num-expert-trajectories 7&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 7&
# done

# Correction: Basic motion shape experiments
basic_motions_correct=("DoubleBendedLine")
for shape in "${basic_motions_correct[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/single_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 4&
    $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/single_models --bijection --num-bijection-layers 8 --crate-lb 1.3  --num-expert-trajectories 4&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/single_models --bijection --num-bijection-layers 8 --crate-lb 1.3  --num-expert-trajectories 1&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/single_models --bijection --num-bijection-layers 8 --crate-lb 1.2  --num-expert-trajectories 1&
done

# Correction Multi model shape experiments
# mm_motions_correct=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")
# for shape in "${mm_motions[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 4 --crate-lb 1.0  --num-expert-trajectories 7&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 7&
# done

# Dimension of x
# for shape in "${basic_motions[@]}"; do
#     $python train.py --model-type discrete --device cuda:0  --dim-x 2 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 4 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 16 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 32 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
#     $python train.py --model-type discrete --device cuda:0  --dim-x 128 --total-epochs 50000 --expert lasa --motion-shape $shape --experiment-dir boards/dimx --batch-size 64&
# done


# Contraction rate
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.0  --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.05 --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.1  --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.2  --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.4  --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.5  --num-expert-trajectories 4&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 50000 --experiment-dir boards/crate_disc --crate-lb 1.6  --num-expert-trajectories 4&

# C-rate impossible made possible by bijection
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 15000 --experiment-dir boards/crate_disc --bijection --num-bijection-layers 8 --crate-lb 1.8  --num-expert-trajectories 1&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 15000 --experiment-dir boards/crate_disc --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 1&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --total-epochs 15000 --experiment-dir boards/crate_disc --bijection --num-bijection-layers 8 --crate-lb 1.6  --num-expert-trajectories 1&

# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --dim-v 64 --total-epochs 15000 --experiment-dir boards/crate_disc --crate-lb 1.8  --num-expert-trajectories 1&



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

# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_1 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_2 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_3 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_4 --experiment-dir boards/multi-models --crate-lb 4.6 --batch-size 70 --num-expert-trajectories 7&

# Augmentation vs no augmentation
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --num-expert-trajectories 1 --total-epochs 15000 --experiment-dir boards/nobij &
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --num-expert-trajectories 1 --total-epochs 15000 --experiment-dir boards/nobij  --num-augment-trajectories 100&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --num-expert-trajectories 1 --total-epochs 15000 --experiment-dir boards/bij --bijection&
# $python train.py --model-type discrete --device cuda:0  --dim-x 64 --num-expert-trajectories 1 --total-epochs 15000 --experiment-dir boards/bij --bijection --num-augment-trajectories 100&
