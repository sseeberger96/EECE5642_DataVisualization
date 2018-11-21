import numpy as np
import csv


def readData(filename, stationCoords): 

	with open(filename, newline='') as input_file: 
		reader = csv.DictReader(input_file)
		data = []
		for row in reader: 
			data.append(row)
			start = int(row['start station id'])
			end = int(row['end station id'])
			startLat = float(row['start station latitude'])
			startLong = float(row['start station longitude'])
			endLat = float(row['end station latitude'])
			endLong = float(row['end station longitude'])
			if not start in stationCoords: 
				stationCoords[start] = (startLat, startLong)
			if not end in stationCoords: 
				stationCoords[end] = (endLat, endLong)

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
	print(len(stationIds))

# def listStationCoords():
# 	stationCoords = {}
# 	for trip in data 

def countTrips(data): 
	numTrips = np.zeros((250,250))
	for trip in data: 
		start = int(trip['start station id'])
		end = int(trip['end station id'])
		numTrips[start][end] += 1 

	return numTrips

	# print(numTrips[200][21])

def getRelAngles(stationCoords): 
	relAngles = np.zeros((250,250))
	for stationA in stationCoords:
		# print(stationA)
		for stationB in stationCoords: 
			if not (stationA == stationB): 
				adjLat = (stationCoords[stationB][0] - stationCoords[stationA][0])* (np.pi/180)
				adjLong = (stationCoords[stationB][1] - stationCoords[stationA][1])* (np.pi/180)
				latARad = stationCoords[stationA][0]* (np.pi/180)
				latBRad = stationCoords[stationB][0]* (np.pi/180)
				longARad = stationCoords[stationA][1]* (np.pi/180)
				longBRad = stationCoords[stationB][1]* (np.pi/180)

				y = np.sin(adjLong)*np.cos(latBRad)
				x = np.cos(latARad)*np.sin(latBRad) - np.sin(latARad)*np.cos(latBRad)*np.cos(adjLong)

				angle = np.arctan2(y,x)*(180/np.pi)

				if angle < 0:
					angle = 360 + angle

				relAngles[stationA][stationB] = angle






				# angle = np.arctan2(adjLat,adjLong)*(180/np.pi)
				# if angle < 0:
				# 	angle = 360 + angle
				# relAngles[stationA][stationB] = angle


	return relAngles










			



if __name__ == '__main__':

	stationCoords = {}
	# print(stationCoords)
	monthData = readData("2017_Data/201701-hubway-tripdata.csv", stationCoords)

	# print(stationCoords)
	# print(len(stationCoords))
	# print(stationCoords[67])
	# print(stationCoords[67][1])

	# print(len(monthData))
	# print(monthData[0])
	# print(monthData[0]['start station id'])

	# searchStationIds(monthData)
	countTrips(monthData)


	angles = getRelAngles(stationCoords)
	print(angles[12][5])
	# print(np.arctan2(0,1)*(180/np.pi))

