#!/bin/bash

python=/isaac-sim/python.sh
basic_motions=("GShape" "DoubleBendedLine" "PShape" "Angle" "Sine" "Worm" "Snake" "NShape")
mm_motions=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")

# Switch to main dir
cd .. || { echo "Failed to change directory"; exit 1; }
echo "Current working directory: $(pwd)"

# Useful command for killing all the background processes
# ps aux | grep 'nnds_training.py' | awk '{print $2}' | xargs kill -9

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
# basic_motions_correct=("Angle")
# for shape in "${basic_motions_correct[@]}"; do
    # echo %%% Running experiments for $shape motion %%%
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir results/corrections/ --bijection --num-bijection-layers 8 --crate-lb 2.0  --num-expert-trajectories 1&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir results/corrections/ --bijection --num-bijection-layers 8 --crate-lb 1.8  --num-expert-trajectories 1&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/ --bijection --num-bijection-layers 8 --crate-lb 1.1  --num-expert-trajectories 4&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/ --num-expert-trajectories 4&
    # $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 30000 --motion-shape $shape --experiment-dir results/corrections/ --bijection --num-bijection-layers 8  --num-expert-trajectories 4&
# done

# Correction Multi model shape experiments
# mm_motions_correct=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")
# for shape in "${mm_motions[@]}"; do
#     echo %%% Running experiments for $shape motion %%%
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 4 --crate-lb 1.0  --num-expert-trajectories 7&
#     $python train.py --model-type discrete  --device cuda:0  --dim-x 64  --total-epochs 15000 --motion-shape $shape --experiment-dir results/lasa_motions/multi_models --bijection --num-bijection-layers 8 --crate-lb 1.5  --num-expert-trajectories 7&
# done

# Dimension of x
basic_motions_correct=("Worm" "Sine")
for shape in "${basic_motions_correct[@]}"; do
    $python train.py --model-type discrete --device cuda:0  --dim-x 2 --total-epochs 15000     --motion-shape $shape --experiment-dir results/dimx --batch-size 64 --num-expert-trajectories 4&
    $python train.py --model-type discrete --device cuda:0  --dim-x 16 --total-epochs 15000    --motion-shape $shape --experiment-dir results/dimx --batch-size 64 --num-expert-trajectories 4&
    $python train.py --model-type discrete --device cuda:0  --dim-x 128 --total-epochs 15000   --motion-shape $shape --experiment-dir results/dimx --batch-size 64 --num-expert-trajectories 4&
    $python train.py --model-type discrete --device cuda:0  --dim-x 1024 --total-epochs 15000  --motion-shape $shape --experiment-dir results/dimx --batch-size 64 --num-expert-trajectories 4&
done


# Ablation: effect of bijection
basic_motions_correct=("CShape" "Sine")
for shape in "${basic_motions_correct[@]}"; do
    $python train.py --model-type discrete --motion-shape $shape --device cuda:0  --dim-x 64 --total-epochs 15000 --experiment-dir boards/crate_disc --bijection --num-bijection-layers 8 --crate-lb 1.8  --num-expert-trajectories 1&
    $python train.py --model-type discrete --motion-shape $shape --device cuda:0  --dim-x 64 --total-epochs 15000 --experiment-dir boards/crate_disc --crate-lb 1.8  --num-expert-trajectories 1&
done