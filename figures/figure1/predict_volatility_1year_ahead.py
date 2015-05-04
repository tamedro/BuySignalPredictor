from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows
    
    
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
    for ii in range(day+252, len(rows) - num_days):
        features = []
        for jj in range(num_days):
            day_index = ii + jj
            features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8]), float(rows[day_index][9])]
        feature_sets += [features]
        value_sets += [float(rows[ii-252][9])]
        value_sets_index.append([ii-252])
        
    # fit
    regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
    regr.fit(feature_sets, value_sets)
    #print "pregr.
    
    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj    +252    
        features += [float(rows[day_index][1]), float(rows[day_index][2]), float(rows[day_index][3]), float(rows[day_index][5]), float(rows[day_index][7]), float(rows[day_index][8]), float(rows[day_index][9])]
        
    return regr.predict(features)



# read in csv file
rows = read_csv('ge-3.csv')

for ii in range(10):
    predicted = predict_volatility_1year_ahead(rows, ii * 63 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[ii*63][9])
    print "prediction for day:", ii*63, "predicted:", predicted, "actual:", actual, "error:", (actual-predicted)**2


print "Paste this into a spreadsheet to plot:"
print "day, error (actual-predicted)^2"
for ii in range(12000):
    predicted = predict_volatility_1year_ahead(rows, ii * 1 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[ii*1][9])
    print ii*1,',', (actual-predicted)**2

