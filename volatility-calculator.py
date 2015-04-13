from numpy import mean
from numpy import array
import math

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

def calculate_volatility(data, time_interval, file_num):
    """
    calculates volatilites
    """
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
            write_results(new_row, file_num)

def write_results(row, file_num):
    import csv
    file_name = 'AFL-original-'
    file_name += `file_num`
    file_name += '.csv'
    with open(file_name, 'a') as csvfile:
        #fieldNames = ['Date','Open','High','Low','Close','Volume','Adj Close','Weekly LogVol','Monthly LogVol','Yearly LogVol']
        writer = csv.writer(csvfile)
        writer.writerow(row)

def read_csv(filename):
    import csv
    with open(filename) as f: rows=[tuple(row) for row in csv.reader(f)]
    print rows[0]       # print field names
    return rows[1:]     # remove field names and return just data 
    
rows = read_csv('AFL-original.csv')
data = Data(rows)
calculate_volatility(data, 7, 1)
rows = read_csv('AFL-original-1.csv')
data = Data(rows)
calculate_volatility(data, 31, 2)
rows = read_csv('AFL-original-2.csv')
data = Data(rows)
calculate_volatility(data, 365, 3)
