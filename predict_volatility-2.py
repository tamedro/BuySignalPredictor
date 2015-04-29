from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model

import argparse
import sys


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows[1:]     # remove field names and return just data 
    
def compile_features_and_values(rows, date_row):   
    num_days = 10
    feature_sets = []
    value_sets = []
    for ii in range(date_row, len(rows) - num_days):
        features = []
        for jj in range(num_days):
            day_index = ii + jj
            features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8])]
        feature_sets += [features]
        value_sets += [float(rows[ii][9])]
    return feature_sets, value_sets    
    
def predict(regr, rows, day):
    num_days = 10
    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj        
        features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8])]
    return regr.predict(features)

def get_real_values(rows, num_predictions):
    real_values = []
    for ii in range(num_predictions):
        real_values.append(float(rows[ii][9]))
    return real_values

def get_accuracy(predictions, real_values):
    diff_sum = 0
    for ii in range (len(predictions)):
        diff = predictions[ii] - real_values[ii]
        diff_squared = diff * diff
        diff_sum = diff_sum + diff_squared
    return diff_sum / len(predictions)

def get_date_row(rows, predict_date):
    for ii in range (len(rows)):
        if rows[ii][0] == predict_date:
            return ii
    return -1

argparser = argparse.ArgumentParser()
argparser.add_argument("--sym", help="stock symbol",
                        type=str, default='ge', required=False)
argparser.add_argument("--predictDate", help="day you would like to predict volatilty",
                        type=str, default='2013-04-09', required=False)
args = argparser.parse_args()

rows = read_csv(args.sym +'-4.csv')
predict_date = args.predictDate
date_row = get_date_row(rows, predict_date)
if date_row < 365:
    print("predict date must be at least 251 days into file")
    sys.exit()
features, mpg = compile_features_and_values(rows, date_row)
regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
regr.fit(features, mpg)
real_values = []
real_values.append(float(rows[date_row - 365][9]))
predictions = []
predictions.append(predict(regr, rows, date_row))
print "real_values: ", real_values
print "predictions: ", predictions
error = get_accuracy(predictions, real_values)
print "error: ", error


