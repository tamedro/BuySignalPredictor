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
    
def compile_features_and_values(rows, date_row, regression_days, predictor):   
    feature_sets = []
    value_sets = []
    if (predictor):
        extra = 1
    else: 
        extra = 0 
    for ii in range(date_row - extra):
        features = []
        for jj in range(regression_days):
            day_index = ii - extra + jj
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
        value_sets += [float(rows[ii][9])]
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

def get_symbols(test):
    '''get rows for each symbol'''
    symbols = []
    if (test):
        filename = 'test-symbols.txt'
    else:
        filename = 'train-symbols.txt'
    f = open(filename, 'r')
    for line in f:
        if line:
            symbols.append(line.rstrip())
    return symbols

def get_rows(symbols):
    '''Get rows'''
    rows = []
    for sym in symbols:
        print(sym)
        filename = sym +'-processed.csv'
        rows.append(read_csv("./DATA/processed_csvs/" + filename))
    return rows

def get_rows_for_file(symbol):
    filename = sym +'-processed.csv'
    return read_csv("./DATA/processed_csvs/" + filename)

def make_symbol_daterow_map(symbols, rows, predict_date):
    date_rows = []
    for ii in range(len(rows)):
        date_rows.append(get_date_row(rows[ii], predict_date))
        if date_rows[ii] < 252 or (date_rows[ii] + regression_days) > len(rows[ii]):
            print("predict date must be at least 251 days into file", ii)
            sys.exit()
    return date_rows

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
argparser.add_argument("--fit_intercept", help="fit_intercept",
                        type=bool, default=False, required=False)
argparser.add_argument("--normalize", help="fit_intercept",
                        type=bool, default=False, required=False)
argparser.add_argument("--max_iter", help="fit_intercept",
                        type=int, default=10000000, required=False)
argparser.add_argument("--l1_ratio", help="fit_intercept",
                        type=float, default=0.5, required=False)
args = argparser.parse_args()
regression_days = args.regressionDays

real_values = []
predictions = []
train_symbols = get_symbols(False)
for sym in train_symbols:
    print(sym)
    rows = get_rows_for_file(sym)
    all_features = []
    all_mpg = []
    #for ii in range(len(rows) - regression_days - 1):
    for ii in range(20):
        features, mpg = compile_features_and_values(rows, len(rows) - ii - regression_days, regression_days, False)
        regr = linear_model.Lasso(alpha=args.alpha,fit_intercept=args.fit_intercept,normalize=args.normalize,max_iter=args.max_iter)
        regr.fit(features, mpg)
        real_values.append(float(rows[len(rows) - ii - regression_days][9]))
        test_features, test_values = compile_features_and_values(rows, len(rows) - ii - regression_days - 1, regression_days, True)
        predictions.append(regr.predict(features)[0])

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
f.write('fit_intercept: ' + str(args.fit_intercept) + '\n')
f.write('l1_ratio: ' + str(args.l1_ratio) + '\n')
f.write('normalize: ' + str(args.normalize) + '\n')
f.write('max-iter: ' + str(args.max_iter) + '\n')
f.write('Training symbols: ' + str(train_symbols) + '\n')
f.write('Real volatilties: ' + str(real_values) + '\n')
f.write('Predicted Volatilities: ' + str(predictions) + '\n')
f.write('Error: ' + str(error) + '\n')
f.write('\n\n')
f.close
