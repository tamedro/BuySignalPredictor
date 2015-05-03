# download_yahoo_histories.py

import urllib
import shutil
import csv

with open('DATA/sp500.csv') as f: rows=[tuple(row) for row in csv.reader(f)]

for i in range(1, len(rows)):
    symbol = rows[i][0]
    saveFilename = 'DATA/original_csvs/' + symbol + '.csv'

    # s=<symbol>
    # a=<end_date_month - 1>
    # b=<end_date_day>
    # c=<end_date_year>
    # d=<start_date_month - 1>
    # e=<start_date_day>
    # f=<start_date_year>
    # g=<d|w|m>   (daily, weekly, or monthly)

    # Download all of the available history up up through May 3, 2015
    url = "http://real-chart.finance.yahoo.com/table.csv?s={}&d=4&e=3&f=2015&g=d&a=0&b=2&c=1962&ignore=.csv".format(symbol)
    filename = urllib.urlretrieve(url, filename=saveFilename)
    print "Saved {}...".format(saveFilename)

