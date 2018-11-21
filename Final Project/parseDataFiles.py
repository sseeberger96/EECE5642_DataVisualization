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


def getRelAngles(stationCoords): 
	relAngles = -1*np.ones((250,250))
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
				angle = -1*(angle - 90)

				if angle < 0:
					angle = 360 + angle

				relAngles[stationA][stationB] = angle
			else: 
				relAngles[stationA][stationB] = 0

	return relAngles

def getQuantityFlowVectors(stationCoords, relAngles, numTrips, numSlices):
	angleInc = 360/numSlices
	flowVectors = np.zeros((250,numSlices)).tolist()

	for stationA in stationCoords: 
		minAngle = 0
		maxAngle = angleInc
		stationAngles = relAngles[stationA][:]
		# if (stationA==12):
		# 	# print(relAngles[stationA][:])
		# 	print(minAngle)
		# 	print(maxAngle)
		for i in range(numSlices):
			flowCount = 0
			stationsInRange = np.where((stationAngles >= minAngle) & (stationAngles < maxAngle))
			stationsInRange = stationsInRange[0].tolist()
			anglesInRange =  stationAngles[(stationAngles >= minAngle) & (stationAngles < maxAngle)]

			if stationsInRange:
				for stationB in stationsInRange:
					flowCount += numTrips[stationA][stationB] 
				avgAngle = np.average(anglesInRange)
			else: 
				avgAngle = 0 

			flowVectors[stationA][i] = (flowCount, avgAngle)
			minAngle += angleInc
			maxAngle += angleInc

			# if (stationA==12 and i==0):
			# 	print(stationsInRange)
			# 	print(anglesInRange)

	return flowVectors


			



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
	numTrips = countTrips(monthData)


	relAngles = getRelAngles(stationCoords)
	# print(relAngles[12][27])
	flowVectors = getQuantityFlowVectors(stationCoords, relAngles, numTrips, 4)
	print(flowVectors[12][:])
	# print(numTrips[12][:])
	# print(relAngles[12][:])
	# print(np.arctan2(0,1)*(180/np.pi))

