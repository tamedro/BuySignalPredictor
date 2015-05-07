import numpy as np

from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin    

import argparse
import sys

import timeit

class_instance = 0
class CustomClassifier(BaseEstimator, ClassifierMixin):
     """Predicts the majority class of its training data."""
     def __init__(self):
         global class_instance
         class_instance += 1
         self.instance = class_instance
         #print "instance:", self.instance
         
     def __del__(self):
         global class_instance
         class_instance -= 1
        
     def fit(self, X, y):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 1:     
             return self
         # 2+ Adaboost iteration: use linera regreession as a weak learner
         else:
             self.regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
             self.regr.fit(X, y)
     
     def predict(self, X):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 1:
             return X[5]   # return 6th element of feature vector (which is the current volatility) 
         # 2+ Adaboost iteration: use linera regreession as a weak learner    
         else:
             return self.regr.predict(X)
             
def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    f.close()
    return rows[1:]     # remove field names and return just data 
    
def compile_features_and_values(rows, date_row, regression_days):   
    feature_sets = []
    value_sets = []
    for ii in range(date_row, len(rows) - regression_days):
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

def predict_volatility_1year_ahead(rows, day):
    """
    SUMMARY: Predict volatility 1 year into the future
    ALGORITHM:
      a) The predictor will train on all data up to exactly 1 year (252 trading days) before `day`
      b) The newest 10 days up to and including `day` will be used as the feature vector for the prediction
         i.e. if day = 0, then the feature vector for prediction will consist of days (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
              if day = 10, then the feature vector for predictor input will be days (10, 11, 12, 13, 14, 15, 16, 17, 19)
    INPUT: minimum of (1 year + 10 days) of data before `day` (newest data is day=0)
  
    """

    #num_days = 10
    num_days = 10

    # enforce that `day` is in the required range
    assert len(rows) >= 252+num_days + day, 'You need to have AT LEAST 252+%d rows AFTER the day index. See predict_volatility_1year_ahead() for details.' % num_days
    assert day >= 0

    # compile features (X) and values (Y) 
    feature_sets = []
    value_sets = []; value_sets_index = []
    for ii in range(num_days+252, len(rows) - num_days):
        features = []
        for jj in range(num_days):
            day_index = ii + jj
            features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8])]
        feature_sets += [features]
        value_sets += [float(rows[ii-252][9])]
        value_sets_index.append([ii-252])
        
    rng = np.random.RandomState(1)
    regr = AdaBoostRegressor(CustomClassifier(), n_estimators=4, random_state=rng)
    regr.fit(feature_sets, value_sets)

    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj    +252    
        features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8])]
        
    return float(regr.predict([features]))

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

for ii in range(10):
    predicted = predict_volatility_1year_ahead(rows[0], ii * 63 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[0][ii*63][9])
    print "prediction for day:", ii*63, "predicted:", predicted, "actual:", actual, "error:", (predicted-actual)**2

for ii in range(20):
    predicted = predict_volatility_1year_ahead(rows[0], ii * 63 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[0][ii*63][9])
    print ii*63,',', (predicted-actual)**2


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
f.write('Training symbols: ' + str(symbols) + '\n')
f.write('Real volatilties: ' + str(real_values) + '\n')
f.write('Predicted Volatilities: ' + str(predictions) + '\n')
f.write('Error: ' + str(error) + '\n')
f.write('\n\n')
f.close
