#!/bin/bash

python=/isaac-sim/python.sh
basic_motions=("GShape" "DoubleBendedLine" "PShape" "Angle" "Sine" "Worm" "Snake" "NShape")
mm_motions=("Multi_Models_1" "Multi_Models_2" "Multi_Models_3" "Multi_Models_4")

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Sine --experiment-dir boards/cont --crate-lb 14.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Sine --experiment-dir boards/cont --crate-lb 12.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&
$python train.py --model-type continuous --device cuda:0  --dim-x 32 --total-epochs 2000 --motion-shape Sine --experiment-dir boards/cont --crate-lb 10.6 --bijection --num-bijection-layers 8 --num-expert-trajectories 4&

