#!/bin/bash
clear
echo "Starting..."
for iter in 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10
do
	echo "Running alpha $iter"
	python predict_volatility-2.py --regressionDays 10 --alpha $iter
done								