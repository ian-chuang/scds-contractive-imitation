#!/bin/bash

python=/isaac-sim/python.sh
motions=("CShape" "DoubleBendedLine" "PShape" "Angle" "Sine")

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# Iterate over the colors array
for shape in "${motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    # $python train.py --model-type discrete --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0&
    $python train.py --model-type continuous --device cpu --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 1000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0
    # $python train.py --model-type discrete --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.2&
    $python train.py --model-type continuous --device cpu --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 1000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.2
    # $python train.py --model-type discrete --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.5&
    # $python train.py --model-type continuous --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --dim-v 2 --total-epochs 1000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.5&
done
