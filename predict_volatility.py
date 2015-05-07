from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows[1:]     # remove field names and return just data 
    
def compile_features_and_values(rows):   
    num_days = 10
    feature_sets = []
    value_sets = []
    for ii in range(0, len(rows) - num_days):
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
    features = [float(rows[day][1]), float(rows[day][2]), float(rows[day][3]), float(rows[day][5]), float(rows[day][7]), float(rows[day][8]), float(rows[day][9])]
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

rows = read_csv('ge-4.csv')
features, mpg = compile_features_and_values(rows)
regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
result = regr.fit(features, mpg)
real_values = get_real_values(rows, 10)
predictions = []
for day in range(10):
    predictions.append(predict(regr, rows, day))
error = get_accuracy(predictions, real_values)
print error
print predictions

