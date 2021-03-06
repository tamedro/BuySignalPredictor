from numpy import mean
from numpy import array

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin    

import sklearn
from sklearn import linear_model

import timeit


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    return rows

def get_coefficients(coeffs_vec, num_features, regression_days):
    coeffs = []
    for ii in range(num_features):
        coeffs.append([])
        for jj in range(regression_days):
            coeffs[ii].append(coeffs_vec[ii + jj*num_features])
    return coeffs

class_instance = 0
class CustomClassifier(BaseEstimator, ClassifierMixin):
     """Predicts the majority class of its training data."""
     def __init__(self):
         global class_instance
         class_instance += 1
         self.instance = class_instance
         
     def __del__(self):
         global class_instance
         class_instance -= 1
        
     def fit(self, X, y):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 2:     
             return self
         # 2+ Adaboost iteration: use linera regreession as a weak learner
         else:
             self.regr = linear_model.Lasso(alpha=0.001,fit_intercept=False,normalize=False,max_iter=10)
             self.regr.fit(X, y)

     def predict(self, X):
         # 1st Adaboost iteration: just return the current volatility
         if self.instance <= 2:
             return X[:,7]   # return 6th element of feature vector (which is the current volatility) 
         # 2+ Adaboost iteration: use linera regreession as a weak learner    
         else:
             return self.regr.predict(X)

    
def predict_volatility_1year_ahead(rows, day, num_days):
    """
    SUMMARY: Predict volatility 1 year into the future
    ALGORITHM:
      a) The predictor will train on all data up to exactly 1 year (252 trading days) before `day`
      b) The newest 10 days up to and including `day` will be used as the feature vector for the prediction
         i.e. if day = 0, then the feature vector for prediction will consist of days (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
              if day = 10, then the feature vector for predictor input will be days (10, 11, 12, 13, 14, 15, 16, 17, 19)
    INPUT: minimum of (1 year + 10 days) of data before `day` (newest data is day=0)
  
    """

    '''enforce that `day` is in the required range'''
    assert len(rows) >= 252+num_days + day, 'You need to have AT LEAST 252+%d rows AFTER the day index. See predict_volatility_1year_ahead() for details.' % num_days
    assert day >= 0

    '''Compile features for fitting'''
    feature_sets = []
    value_sets = []; value_sets_index = []
    for ii in range(day+num_days+252, len(rows) - num_days):
        features = []
        for jj in range(num_days):
            day_index = ii + jj
            features += [
        	float(rows[day_index][1]), 
        	float(rows[day_index][2]), 
        	float(rows[day_index][3]), 
        	float(rows[day_index][5]),
        	float(rows[day_index][6]), 
        	float(rows[day_index][7]), 
        	float(rows[day_index][8]),
        	float(rows[day_index][9]), 
        	float(rows[day_index][10]),
        	float(rows[day_index][11]),
        	float(rows[day_index][12]),
        	float(rows[day_index][13]),
        	float(rows[day_index][14]),
        	float(rows[day_index][15]),
        	float(rows[day_index][16]),
        	float(rows[day_index][17])
            ]
            #print("issue here: " + str(rows[day_index][0]))
        feature_sets += [features]
        value_sets += [float(rows[ii-252][9])]
        value_sets_index.append([ii-252])

    '''Create Regressor and fit'''
    num_features = 16
    rng = np.random.RandomState(1)
    regr = AdaBoostRegressor(CustomClassifier(), n_estimators=2, random_state=rng)
    regr.fit(feature_sets, value_sets)
    regr_coeffs = regr.estimators_[1].regr.coef_
    #print(regr_coeffs)
    split_coeffs = get_coefficients(regr_coeffs, num_features, regression_days)
    #print(split_coeffs)
    coeffs = []
    for ii in range(num_features):
    	coeffs.append(sum(split_coeffs[ii]) / float(len(split_coeffs)))

    '''Get prediction features'''
    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj + 252    
        features += [
        float(rows[day_index][1]), 
        float(rows[day_index][2]), 
        float(rows[day_index][3]), 
        float(rows[day_index][5]),
        float(rows[day_index][6]), 
        float(rows[day_index][7]), 
        float(rows[day_index][8]),
        float(rows[day_index][9]), 
        float(rows[day_index][10]),
        float(rows[day_index][11]),
        float(rows[day_index][12]),
        float(rows[day_index][13]),
        float(rows[day_index][14]),
        float(rows[day_index][15]),
        float(rows[day_index][16]),
        float(rows[day_index][17])
        ]
        
    return float(regr.predict([features])) , split_coeffs


rows = read_csv('AAPL-processed.csv')

'''Begin Main'''
print('Starting...')
regression_days = 10
num_features = 16
    
#log = []
worst_coeffs = [[], [], [], [], []]
worst_errors = [0, 0, 0, 0, 0]
best_coeffs = [[], [], [], [], []]
best_errors = [100, 100, 100, 100, 100,]
for ii in range(6250):
#for ii in range(10):
    print('iteration ' + str(ii) + ' of 6250') 
    predicted, coeffs = predict_volatility_1year_ahead(rows, ii, regression_days)
    actual    = float(rows[ii + 252][9])
    error = (predicted-actual)**2
    for ii in range(len(best_errors)):
    	if error < best_errors[ii]:
    		best_coeffs[ii] = coeffs
    		best_errors[ii] = error
    		break
    for ii in range(len(worst_errors)): 
    	if error > worst_errors[ii]:
    		worst_coeffs[ii] = coeffs 
    		worst_errors[ii] = error
    		break

avg_best_error = sum(best_errors) / float(len(best_errors))
avg_worst_error = sum(worst_errors) / float(len(worst_errors))
worst = []
worst_avg = []
for ii in range(num_features):
	feat = []
	for jj in range(len(worst_coeffs)):
		feat.append(sum(worst_coeffs[jj][ii]) / float(len(worst_coeffs[jj][ii])))
	worst.append(feat)
for ii in range(len(worst)):
	worst_avg.append(sum(worst[ii]) / float(len(worst[ii])))
best = []
best_avg = []
for ii in range(num_features):
	feat = []
	for jj in range(len(best_coeffs)):
		feat.append(sum(best_coeffs[jj][ii]) / float(len(best_coeffs[jj][ii])))
	best.append(feat)
for ii in range(len(best)):
	best_avg.append(sum(best[ii]) / float(len(best[ii])))
worst_avg_string = ""
best_avg_string = ""
for ii in range(len(worst_avg)):
	worst_avg_string += str(worst_avg[ii]) + ',' 
for ii in range(len(best_avg)):
	best_avg_string += str(best_avg[ii]) + ','
print "worst error was " + str(avg_worst_error) + " with coeffs: "
print worst_avg_string
print "best error was " + str(avg_best_error) + " with coeffs: "
print best_avg_string
#print best_coeffs
'''record results'''
f  = open('log.csv', 'a')
f.write(worst_avg_string + '\n')
f.write(best_avg_string + '\n')
f.close

