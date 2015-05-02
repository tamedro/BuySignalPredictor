#!/bin/bash
clear
echo "Starting..."
for iter in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
	echo "Running ratio $iter"
	python predict_volatility-2.py --regressionDays 10 --l1_ratio $iter
done								