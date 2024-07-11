#!/bin/bash

python=/isaac-sim/python.sh
motions=("CShape" "DoubleBendedLine" "PShape" "SShape" "Angle" "Sine")

# Iterate over the colors array
for shape in "${motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    $python train.py --model-type discrete --device cuda:0 --horizon 20  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape &
    $python train.py --model-type discrete --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape &
    $python train.py --model-type discrete --device cuda:0 --horizon 100 --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape &

    wait

    $python train.py --model-type discrete --device cuda:0 --horizon 20  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 15000 --expert lasa --motion-shape $shape &
    $python train.py --model-type discrete --device cuda:0 --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 15000 --expert lasa --motion-shape $shape &
    $python train.py --model-type discrete --device cuda:0 --horizon 100 --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 15000 --expert lasa --motion-shape $shape &

    wait
done
