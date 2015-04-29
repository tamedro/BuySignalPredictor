from numpy import mean
from numpy import array
import math
import datetime as dt
import argparse

class Data:
    def __init__(self, rows):
        self.rows = rows
        self.date_index = 0
        self.open_index = 1
        self.close_index = 4
        self.high_index = 2
        self.low_index = 3
        self.volume_index = 5
        self.adj_close_index = 6

def write_results(data, symbol):
    import csv
    file_name = symbol + '-'
    file_name += 'processed'
    file_name += '.csv'
    file_name = './DATA/processed_csvs/' + file_name
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for ii in range(len(data.rows)):
            writer.writerow(data.rows[ii])

def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    print rows[0]       # print field names
    return rows[1:]     # remove field names and return just data 

def calculate_volatility(data, time_interval):
    """
    calculates volatilites
    """
    rows = []
    for ii in range(len(data.rows)):
        if ii + time_interval < len(data.rows):
            avg_close = 0            
            deviation = 0
            st_dev_sum = 0
            returns_sum = 0
            st_dev_list = []
            for jj in range(time_interval):
                returns =  (float(data.rows[ii + jj][data.close_index]) / (float(data.rows[ii + jj + 1][data.close_index]))) - 1
                returns_sum = returns_sum + returns
            avg_close = returns_sum / time_interval
            for jj in range(time_interval):
                deviation =  ((float(data.rows[ii + jj][data.close_index]) / (float(data.rows[ii + jj + 1][data.close_index]))) - 1) - avg_close
                deviation_sq = deviation * deviation
                st_dev_list.append(deviation_sq)
            for jj in range (len(st_dev_list)):
                st_dev_sum = st_dev_sum + st_dev_list[jj]
            st_dev = st_dev_sum / time_interval
            volatility = math.sqrt(st_dev)
            log_vol = math.log(volatility)
            new_row = []
            for jj in range (len(data.rows[ii])):
                new_row.append(data.rows[ii][jj])
            new_row.append(log_vol)
            rows.append(new_row)
    return rows

def calculate_day_of_week(data):
    num_columns = len(data.rows[0])
    rows = []
    for ii in range(len(data.rows) -1 ):
        date_string = data.rows[ii][0]
        date_array = date_string.split("-")
        today = dt.datetime(int(date_array[0]),int(date_array[1]), int(date_array[2]))
        new_row = []
        for jj in range(num_columns):
            new_row.append(data.rows[ii][jj])
        new_row.append(int(today.weekday()))
        rows.append(new_row)
    return rows

'''Parse arguments'''
argparser = argparse.ArgumentParser()
argparser.add_argument("--sym", help="stock symbol",
                        type=str, default='ge', required=False)
argparser.add_argument("--predictDate", help="day you would like to predict volatilty",
                        type=str, default='2013-04-09', required=False)
args = argparser.parse_args()
read_path = './DATA/original_csvs/'
write_path = './DATA/processed_csvs/'

symbols = []
f = open('symbols.txt', 'r')
for line in f:
    symbols.append(line.rstrip())

for symbol in symbols:
    rows = read_csv(read_path + symbol + '.csv')
    data = Data(rows)
    data.rows = calculate_volatility(data, 7,)
    data.rows = calculate_volatility(data, 31,)
    data.rows = calculate_volatility(data, 365,)
    data.rows = calculate_day_of_week(data)
    write_results(data, symbol)
