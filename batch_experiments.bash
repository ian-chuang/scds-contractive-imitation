#!/bin/bash

python=/isaac-sim/python.sh
motions=("CShape" "DoubleBendedLine" "PShape" "SShape" "Angle")

# Iterate over the colors array
for shape in "${motions[@]}"; do
    echo %%% Running experiments for $shape motion %%%
    $python ren_policy_simple.py --device cpu --horizon 20  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape
    $python ren_policy_simple.py --device cpu --horizon 50  --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape
    $python ren_policy_simple.py --device cpu --horizon 100 --dim-x 8 --dim-in 2 --dim-out 2 --l-hidden 2 --total-epochs 30000 --expert lasa --motion-shape $shape

done
