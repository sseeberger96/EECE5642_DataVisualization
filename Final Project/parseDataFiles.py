import numpy as np
import csv


def readData(filename): 

	with open(filename, newline='') as input_file: 
		reader = csv.DictReader(input_file)
		data = []
		for row in reader: 
			data.append(row)

	return data


			



if __name__ == '__main__':
	monthData = readData("2017_Data/201701-hubway-tripdata.csv")

	print(len(monthData))
	print(monthData[0])
	print(monthData[0]['start station id'])