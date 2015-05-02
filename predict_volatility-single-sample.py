from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model

import argparse
import sys

import timeit


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    f.close()
    return rows[1:]     # remove field names and return just data 
    
def compile_features_and_values(rows, date_row, regression_days):   
    feature_sets = []
    value_sets = []
    features = []
    for jj in range(regression_days):
        day_index = date_row + jj
        features += [
        float(rows[day_index][1]), 
        float(rows[day_index][2]), 
        float(rows[day_index][3]), 
        float(rows[day_index][5]),
        float(rows[day_index][6]), 
        float(rows[day_index][7]), 
        float(rows[day_index][8]), 
        float(rows[day_index][10]),
        float(rows[day_index][11]),
        float(rows[day_index][12]),
        float(rows[day_index][13]),
        float(rows[day_index][14])
        ]
    feature_sets += [features]
    value_sets += [float(rows[date_row][9])]
    return feature_sets, value_sets    
    
def predict(regr, rows, day, regression_days):
    ii = day
    features = []
    for jj in range(regression_days):
        day_index = ii + jj        
        features += [
        float(rows[day_index][1]), 
        float(rows[day_index][2]), 
        float(rows[day_index][3]), 
        float(rows[day_index][5]),
        float(rows[day_index][6]), 
        float(rows[day_index][7]), 
        float(rows[day_index][8]), 
        float(rows[day_index][10]),
        float(rows[day_index][11]),
        float(rows[day_index][12]),
        float(rows[day_index][13]),
        float(rows[day_index][14])
        ]
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

'''parse arguments'''
start = timeit.default_timer()
argparser = argparse.ArgumentParser()
argparser.add_argument("--sym", help="stock symbol",
                        type=str, default='ge', required=False)
argparser.add_argument("--predictDate", help="day you would like to predict volatilty",
                        type=str, default='2013-04-09', required=False)
argparser.add_argument("--regressionDays", help="Amount of days in a regression sample",
                        type=int, default=10, required=False)
argparser.add_argument("--alpha", help="Alpha for model",
                        type=float, default=0.001, required=False)
args = argparser.parse_args()
regression_days = args.regressionDays

'''get rows for each symbol'''
symbols = []
f = open('symbols.txt', 'r')
for line in f:
    symbols.append(line.rstrip())
print symbols
rows = []
for sym in symbols:
    filename = sym +'-processed.csv'
    rows.append(read_csv("./DATA/processed_csvs/" + filename))

'''get predict date row value for each symbol'''
predict_date = args.predictDate
date_rows = []
for ii in range(len(rows)):
    date_rows.append(get_date_row(rows[ii], predict_date))
    if date_rows[ii] < 252 or (date_rows[ii] + regression_days) > len(rows[ii]):
        print("predict date must be at least 251 days into file", ii)
        sys.exit()

'''compile features over all symbols'''
all_features = []
all_mpg = []
for ii in range(len(rows)):
    print("compiling file ", ii + 1, " of ", len(rows), "...")
    features, mpg = compile_features_and_values(rows[ii], date_rows[ii], regression_days)
    all_features += features
    all_mpg += mpg

'''build and fit model'''
print("Fitting...")
regr = linear_model.Ridge(alpha=args.alpha,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
regr.fit(all_features, all_mpg)

'''Make Predictions'''
print("Predicting...")
real_values = []
predictions = []
for ii in range(len(rows)): 
    real_values.append(float(rows[ii][date_rows[ii] - 252][9]))
    predictions.append(predict(regr, rows[ii], date_rows[ii], regression_days))

'''calculate error'''
error = get_accuracy(predictions, real_values)
stop_time = timeit.default_timer()
print("Done")
print "real_values: ", real_values
print "predictions: ", predictions
print "error: ", error


'''record results'''
f  = open('log', 'a')
f.write('***** New Run *****' + '\n')
f.write('Execution time: ' + str(stop_time - start) + 's' + '\n')
f.write('Regressions days:' + str(regression_days) + '\n')
f.write('Alpha: ' + str(args.alpha) + '\n')
f.write('Training symbols: ' + str(symbols) + '\n')
f.write('Real volatilties: ' + str(real_values) + '\n')
f.write('Predicted Volatilities: ' + str(predictions) + '\n')
f.write('Error: ' + str(error) + '\n')
f.write('\n\n')
f.close
