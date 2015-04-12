# predict close price

from numpy import mean
from numpy import array

import sklearn
from sklearn import linear_model


def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    print rows[0]       # print field names
    return rows[1:]     # remove field names and return just data 
    
def compile_features_and_values(rows):   
    num_days = 1
    
    feature_sets = []
    value_sets = []
    for ii in range( len(rows)-num_days ):
        features = []
        for jj in range( num_days ):
            day_index = ii + jj
            #print ii, jj, day_index
        
            # fields: Date,Open,High,Low,Close,Volume,Adj Close
            features += [float(rows[day_index][1]), float(rows[day_index][2])]
        
        feature_sets += [features]
        value_sets += [float(rows[ii][1])]
    return feature_sets, value_sets    
    
def predict(regr, rows, day):
    num_days = 1

    # day = 0 is the most recent day
    ii = day
    features = []
    for jj in range( num_days ):
        day_index = ii + jj        
        # fields: Date,Open,High,Low,Close,Volume,Adj Close
        features += [float(rows[day_index][1]), float(rows[day_index][2])]

    print "volatility for 2007: ", regr.predict(features)
    
    
    

#features, mpg = read_mpg('data.txt')
rows = read_csv('AFLAC-log-volatility.csv')
features, mpg = compile_features_and_values(rows)

#print mpg

# Create linear regression object
#regr = linear_model.LinearRegression()
regr = linear_model.Lasso(alpha=1)   # they call lambda alpha

# Train the model using the training sets
regr.fit(features, mpg)

print regr.coef_
print regr.intercept_

for day in range(1):
    predict(regr, rows, day)

#print "mean(mpg):", mean(mpg)

