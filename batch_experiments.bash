#!/bin/bash

python=/isaac-sim/python.sh
# motions=("CShape" "DoubleBendedLine" "PShape" "Angle" "Sine")
motions=("CShape")

# Useful command for killing all the background processes
# ps aux | grep 'train.py' | awk '{print $2}' | xargs kill -9

# Iterate over different motions shapes
for shape in "${motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    # $python train.py --model-type discrete --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0&
    # $python train.py --model-type continuous --device cpu  --dim-x 8  --total-epochs 10000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    # $python train.py --model-type discrete --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.2&
    # $python train.py --model-type continuous --device cpu  --dim-x 8  --total-epochs 10000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.2 &
    # $python train.py --model-type discrete --device cuda:0  --dim-x 8  --total-epochs 30000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.5&
    # $python train.py --model-type continuous --device cuda:0  --dim-x 8  --total-epochs 1000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.5&
done

for shape in "${motions[@]}"; do
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
    $python train.py --model-type discrete --device cuda:0  --dim-x 8 --total-epochs 40000 --expert lasa --motion-shape $shape --experiment-dir batch --ic-noise-rate 0.0 &
done