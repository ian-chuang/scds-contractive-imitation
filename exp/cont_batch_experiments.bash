#!/bin/bash

python=/isaac-sim/python.sh
basic_motions=("GShape" "DoubleBendedLine" "PShape" "Angle" "Worm" "NShape")
mm_motions=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")

# Switch to main dir
cd .. || { echo "Failed to change directory"; exit 1; }
echo "Current working directory: $(pwd)"

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Trapezoid --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Zshape --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_4 --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_3 --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Leaf_1 --experiment-dir boards/cont --crate-lb 8.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Spoon --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&


# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Trapezoid --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Zshape --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_4 --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Multi_Models_3 --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 7&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Leaf_1 --experiment-dir boards/cont --crate-lb 8.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
# $python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Spoon --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&


# Basic motion shape experiments
for shape in "${basic_motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 12.6  --num-expert-trajectories 4&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 12.6  --num-expert-trajectories 4&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 12.6  --num-expert-trajectories 1&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 12.6  --num-expert-trajectories 1&

    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 5.6  --num-expert-trajectories 4&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 5.6  --num-expert-trajectories 4&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 5.6  --num-expert-trajectories 1&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/single_models --bijection --num-bijection-layers 10 --crate-lb 5.6  --num-expert-trajectories 1&
done

# Multi model shape experiments
for shape in "${mm_motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/multi_models --bijection --num-bijection-layers 10 --crate-lb 12.6  --num-expert-trajectories 7&
    $python train.py --model-type continuous  --device cuda:0  --dim-x 64  --total-epochs 1000 --motion-shape $shape --experiment-dir boards/cont/new/multi_models --bijection --num-bijection-layers 10 --crate-lb 18.6  --num-expert-trajectories 7&
done
