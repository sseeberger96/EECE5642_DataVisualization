import numpy as np
import csv


def readData(filename): 

	with open(filename, newline='') as input_file: 
		reader = csv.DictReader(input_file)
		data = []
		for row in reader: 
			data.append(row)

	return data


def searchStationIds(data): 
	stationIds = []
	for trip in data:
		if trip['start station id'] not in stationIds: 
			stationIds.append(trip['start station id']) 
		if trip['end station id'] not in stationIds: 
			stationIds.append(trip['end station id'])
	stationIds = [int(x) for x in stationIds]
	stationIds.sort()
	print(stationIds)

def countTrips(data): 
	numTrips = np.zeros((250,250))
	for trip in data: 
		start = int(trip['start station id'])
		end = int(trip['end station id'])
		numTrips[start][end] += 1 

	# print(numTrips[200][21])



			



if __name__ == '__main__':
	monthData = readData("2017_Data/201701-hubway-tripdata.csv")

	# print(len(monthData))
	# print(monthData[0])
	print(monthData[0]['start station id'])

	# searchStationIds(monthData)
	countTrips(monthData)