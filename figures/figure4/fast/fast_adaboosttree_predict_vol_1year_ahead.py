from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model

# configuration
#filename = 'ge-3.csv'
filename = 'GE-processed.csv'

num_days = 10
#num_days = 252

feature_columns = [1, 2, 3, 5, 7, 8, 9]
#feature_columns = [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]



def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows




import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin    
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
        
     def fit(self, X, y, sample_weight=array([])):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 2:
             self.y = y     
             return self
         # 2+ Adaboost iteration: use linera regreession as a weak learner
         else:
             self.regr = DecisionTreeRegressor(max_depth=8)
             #self.regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
             self.regr.fit(X, y)
     
     def predict(self, X):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 2:
             return X[:,6]   # return 6th element of feature vector (which is the current volatility) 
         # 2+ Adaboost iteration: use linera regreession as a weak learner    
         else:
             return self.regr.predict(X)

    




def learn_volatility_1year_ahead(regr, rows, day):
    # glabal var: num_days
    
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
            for kk in feature_columns:
                features += [float(rows[day_index][kk])]
          
        feature_sets += [features]
        value_sets += [float(rows[ii-252][9])]
        value_sets_index.append([ii-252])
        
    # fit
    regr.fit(feature_sets, value_sets)
    #print "pregr.
    
    print "Adaboost weights:", regr.estimator_weights_
    
    
    
    
def predict_volatility_1year_ahead(regr, rows, day):
    """
    SUMMARY: Predict volatility 1 year into the future
    ALGORITHM:
      a) The predictor will train on all data up to exactly 1 year (252 trading days) before `day`
      b) The newest 10 days up to and including `day` will be used as the feature vector for the prediction
         i.e. if day = 0, then the feature vector for prediction will consist of days (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
              if day = 10, then the feature vector for predictor input will be days (10, 11, 12, 13, 14, 15, 16, 17, 19)
    INPUT: minimum of (1 year + 10 days) of data before `day` (newest data is day=0)
  
    """

    # global var: num_days

    # enforce that `day` is in the required range
    assert len(rows) >= 252+num_days + day, 'You need to have AT LEAST 252+%d rows AFTER the day index. See predict_volatility_1year_ahead() for details.' % num_days
    assert day >= 0

    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj    +252  
        for kk in feature_columns:
                features += [float(rows[day_index][kk])]  
        
    return float(regr.predict([features]))





# read in csv file
#rows = read_csv('ge-3.csv')
rows = read_csv(filename)


# learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(1)
regr = AdaBoostRegressor(CustomClassifier(), n_estimators=4, random_state=rng)
#regr = linear_model.Lasso(alpha=0.01,fit_intercept=False,normalize=False,max_iter=10000000)   # they call lambda alpha
day = 7000
learn_volatility_1year_ahead(regr, rows, day)

for ii in range(10):
    predicted = predict_volatility_1year_ahead(regr, rows, ii * 63 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[ii*63][9])
    print "prediction for day:", ii*63, "predicted:", predicted, "actual:", actual, "error:", (actual-predicted)**2


print "Paste this into a spreadsheet to plot:"
print "day, error (actual-predicted)^2"
for ii in range(6000):
    predicted = predict_volatility_1year_ahead(regr, rows, ii * 1 + 252)    # 63 days is exactly 1/4 of a trading year
    actual    = float(rows[ii*1][9])
    print ii*1,',', (actual-predicted)**2

