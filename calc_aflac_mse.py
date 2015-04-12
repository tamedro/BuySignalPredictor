# calc_aflac_mse.py

import mse
import csv

def read_csv(filename):
    with open(filename, 'rb') as f:
        rows = [tuple(row) for row in csv.reader(f)]
    return rows

# read actual volatilities
rows = read_csv('aflac_volatility.csv')

# predicted values: use all actual values but the last
p = []
for i in range(0, len(rows) - 1):
    p.append(float(rows[i][1]))

# actual values: all but the first year, which we don't have a prediction for
a = []
for i in range(1, len(rows)):
    a.append(float(rows[i][1]))

mse = mse.compute_mse(p, a)
print "mse = ", mse

