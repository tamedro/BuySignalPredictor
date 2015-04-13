# mse.py
import math

# p - list of predicted values
# a - list of actual values
def compute_mse(p, a):
    if (len(p) != len(a)):
        print "compute_mse(): # of predicted and actual values do not match"
        return

    sum = 0.0;
    for i in range(0, len(p)):
        #pred = math.log10(p[i])
        pred = p[i]
        #actual = math.log10(a[i])
        actual = a[i]
        sum += (math.pow(pred - actual, 2.0))

    return (sum / len(p))


