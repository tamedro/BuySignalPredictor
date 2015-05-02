#!/bin/bash
clear
echo "Starting..."
for iter in 3 5 10 20 50 100 150
do
	echo "Running alpha $iter"
	python predict_volatility-2.py --regressionDays $iter --max_iter 10 --alpha 0.001
done								