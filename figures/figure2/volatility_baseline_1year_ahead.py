from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows
    
    
# read in csv file
rows = read_csv('../../DATA/processed_csvs/AAPL-processed.csv')

#for ii in range(12000):
for ii in range(len(rows)-252):
    predicted = float(rows[ii * 1 + 252][9])
    actual    = float(rows[ii*1][9])
    print ii*1,',', (actual-predicted)**2

