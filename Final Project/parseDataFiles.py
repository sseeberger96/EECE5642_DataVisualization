import numpy as np
import csv
import pickle
import time
import scipy.stats
from sklearn import linear_model

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import googlePlaces
import censusDataAPI


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

def getStationUsage(data):
	stationUsage = {}
	for trip in data: 
		start = int(trip['start station id'])
		end = int(trip['end station id'])
		if not start in stationUsage:
			stationUsage[start] = 0
		if not end in stationUsage: 
			stationUsage[end] = 0
		stationUsage[start] += 1
		stationUsage[end] += 1

	return stationUsage

def getPOIData(stationCoords, poiTypes, radius):
	poiData = {}
	for station in stationCoords:
		print("Processing Station: %d" % station)
		# print(stationCoords[station][0])	
		lat = stationCoords[station][0]
		lng = stationCoords[station][1]
		stationPOIs = []
		for category in poiTypes:
			uniqueLocs = []
			for searchType in category:
				search = googlePlaces.GooglePlace(lat=lat,lng=lng,radius=radius,types=searchType)
				if search['status'] == 'OK':
					for place in search['results']:
					    if not place['reference'] in uniqueLocs:
					    	uniqueLocs.append(place['reference'])
				
			numPlaces = len(uniqueLocs)
			stationPOIs.append((category[0], numPlaces))
		poiData[station] = stationPOIs
	return poiData

def getCensusTracts(stationCoords):
	print("Getting Census Tracts")
	censusTracts = {}
	for station in stationCoords:
		print("Processing Station: %d" % station)
		# print(stationCoords[station][0])	
		lat = stationCoords[station][0]
		lng = stationCoords[station][1]
		search = censusDataAPI.census(lat,lng)
		if search:
			censusTracts[station] = int(search)
		else: 
			while not search: 
				search = censusDataAPI.census(lat,lng)
				time.sleep(0.01)
			censusTracts[station] = int(search)
	return censusTracts

def readHHIncomes(filename):
	print("Getting Household Incomes")
	with open(filename, newline='') as input_file: 
		reader = csv.DictReader(input_file)
		hhIncomeData = {}
		for row in reader: 
			tractID = int(row['GEOID10'])
			if row['MdHHinc_ES'] != '':
				hhIncome = int(row['MdHHinc_ES'])

				hhIncomeData[tractID] = hhIncome
			else: 
				hhIncomeData[tractID] = 0


	return hhIncomeData

def readAgePopData(filename):
	print("Getting Age and Population Data")
	with open(filename, newline='') as input_file: 
		reader = csv.DictReader(input_file)
		agePopData = {}
		for row in reader: 
			tractID = int(row['ct10_id'])
			medAge = float(row['medage10'])
			totPop = int(row['totpop10'])

			agePopData[tractID] = (medAge, totPop)

	return agePopData

def getCensusData(incomeFilename, agePopFilename, stationCoords):
	censusTracts = getCensusTracts(stationCoords)
	hhIncomeData = readHHIncomes(incomeFilename)
	agePopData = readAgePopData(agePopFilename)

	censusData = {}
	for station in stationCoords:
		tractID = censusTracts[station]
		income = hhIncomeData[tractID]
		age = agePopData[tractID][0]
		pop = agePopData[tractID][1]

		if (income != 0) and (age != 0) and (pop != 0):
			censusData[station] = [income, age, pop]

	return censusData


def processData(usageData, poiData, censusData):
	poiDataProcessed = {}
	usageProcessed = {}
	censusDataProcessed = {}
	for station in sorted(censusData):
		poiDataProcessed[station] = poiData[station]
		usageProcessed[station] = usageData[station]
		censusDataProcessed[station] = censusData[station]

	poiData = poiDataProcessed
	usageData = usageProcessed
	censusData = censusDataProcessed

	proc = []
	for station in usageData: 
		proc.append(usageData[station])

	# mean = np.mean(proc)
	# std = np.std(proc)
	# minVal = min(proc)
	# maxVal = max(proc)

	# proc = [(point - mean)/std for point in proc]
	# if (maxVal-minVal) != 0:
	# 	proc = [(point-minVal)/(maxVal-minVal) for point in proc]

	i = 0
	for station in usageData:
		usageData[station] = proc[i]
		i +=1 

	for j in range(len(poiTypes)):
		proc = []
		for station in poiData: 
			proc.append(poiData[station][j][1])

		# mean = np.mean(proc)
		# std = np.std(proc)
		# minVal = min(proc)
		# maxVal = max(proc)

		# proc = [(point - mean)/std for point in proc]
		# proc = [(point-minVal)/(maxVal-minVal) for point in proc]

		i = 0
		for station in poiData:
			catName = poiData[station][j][0]
			poiData[station][j] = (catName, proc[i])
			i +=1 

	for k in range(3):
		proc = []
		for station in censusData: 
			proc.append(censusData[station][k])

		# mean = np.mean(proc)
		# std = np.std(proc)
		# minVal = min(proc)
		# maxVal = max(proc)

		# proc = [(point - mean)/std for point in proc]
		# proc = [(point-minVal)/(maxVal-minVal) for point in proc]

		i = 0
		for station in censusData:
			censusData[station][k] = proc[i]
			i +=1 

	return usageData, poiData, censusData



def calculateSpearmanPOI(usageData, poiData):
	spearmanPOIs = {}

	statUsages = []
	for station in usageData: 
		statUsages.append(usageData[station])

	statUsages = scipy.stats.rankdata(statUsages)

	statUseMean = np.mean(statUsages)


	for j in range(len(poiTypes)):
		pois = []
		for station in poiData: 
			pois.append(poiData[station][j][1])

		pois = scipy.stats.rankdata(pois)

		poiMean = np.mean(pois)

		numerator = 0 
		denom1 = 0 
		denom2 = 0
		for i in range(len(statUsages)):
			numerator += (statUsages[i] - statUseMean)*(pois[i] - poiMean)
			denom1 += (statUsages[i] - statUseMean)**2
			denom2 += (pois[i] - poiMean)**2

		denom1 = np.sqrt(denom1)
		denom2 = np.sqrt(denom2)
		denominator = denom1*denom2

		rho = numerator/denominator

		spearmanPOIs[poiData[67][j][0]] = rho 

	return spearmanPOIs

def filterPOIs(spearmanPOIs, poiThresh):
	filteredPOIs = {}
	for spear in spearmanPOIs:
		if spearmanPOIs[spear] >= poiThresh: 
			filteredPOIs[spear] = spearmanPOIs[spear]

	return filteredPOIs



def processDataForRegression(usageData, poiData, censusData, filteredPOIs):
	statUsages = []
	for station in usageData: 
		statUsages.append(usageData[station])

	pois = []
	for j in range(len(poiTypes)):
		poiCat = []
		for station in poiData: 
			if poiData[station][j][0] in filteredPOIs.keys():
				poiCat.append(poiData[station][j][1])
		if poiCat:
			pois.append(poiCat)


	income = []
	age = []
	pop = []
	for station in censusData: 
		income.append(censusData[station][0])
		age.append(censusData[station][1])
		pop.append(censusData[station][2])

	target = np.array(statUsages)
	target = np.transpose(target)

	features = []
	for point in pois: 
		features.append(point)
	features.append(income)
	features.append(age)
	features.append(pop)

	features = np.array(features)
	features = np.transpose(features)
	
	return target, features


def generateLinearRegression(target, features):
	regression = linear_model.LinearRegression()
	regression.fit(features, target)

	return regression








	





			



if __name__ == '__main__':

	stationCoords = {}
	poiTypes = [['airport', 'train_station', 'transit_station'], ['aquarium', 'art_gallery', 'stadium', 'museum', 'zoo'], ['bakery', 'cafe'], ['church','hindu_temple', 'mosque', 'synagogue'], 
				['fire_station', 'police'], ['clothing_store', 'department_store', 'electronics_store', 'shopping_mall', 'home_goods_store', 'bicycle_store', 'shoe_store'], 
				['subway_station', 'bus_station', 'taxi_stand'], ['convenience_store', 'pharmacy', 'liquor_store', 'supermarket'], ['local_government_office', 'city_hall'], 
				['restaurant', 'bar', 'meal_delivery', 'meal_takeaway'], ['school', 'university'], ['hospital', 'doctor'], ['atm'], ['bank'], ['gas_station'], ['gym'], ['lawyer'], ['library'], ['lodging'], 
				['movie_theater'], ['night_club'], ['park'], ['parking'], ['post_office']]
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

	usageData = getStationUsage(monthData)
	# print(usageData)


	relAngles = getRelAngles(stationCoords)
	# print(relAngles[12][27])
	flowVectors = getQuantityFlowVectors(stationCoords, relAngles, numTrips, 4)
	# print(flowVectors[12][:])
	# print(numTrips[12][:])
	# print(relAngles[12][:])
	# print(np.arctan2(0,1)*(180/np.pi))

	s = {67: (42.328539, -71.100815)}
	p = [['bakery', 'cafe'], ['school', 'university']]

	# poiData = getPOIData(stationCoords, poiTypes, 250)
	# poiData = getPOIData(s, p, 100)
	# pickle.dump(poiData, open('poiData.p', 'wb'))

	with open('poiData.p', mode='rb') as in_file:
		poiData = pickle.load(in_file)

	# censusData = getCensusData('Census_Data/median_hh_income.csv', 'Census_Data/population_by_gender_age.csv', stationCoords)
	# pickle.dump(censusData, open('censusData.p', 'wb'))

	with open('censusData.p', mode='rb') as in_file:
		censusData = pickle.load(in_file)

	usageData, poiData, censusData = processData(usageData, poiData, censusData)

	spearmanPOIs = calculateSpearmanPOI(usageData, poiData)
	filteredPOIs = filterPOIs(spearmanPOIs, 0.20)
	print(filteredPOIs)
	print(sorted(filteredPOIs.values()))

	target, features = processDataForRegression(usageData, poiData, censusData, filteredPOIs)

	LinearRegression = generateLinearRegression(target, features)



	testCoord = {500: (42.328539, -71.100815)} # Works as a good example
	# testCoord = {500: (42.285715, -71.064233)} # Negative output
	# testCoord = {500: (42.300986, -71.114308)} # Works as a good example
	# testCoord = {500: (42.330681, -71.043363)} # Works as a bad example 


	dummyUsage = {500: 1}

	testPOIData = getPOIData(testCoord, poiTypes, 250)

	testCensusData = getCensusData('Census_Data/median_hh_income.csv', 'Census_Data/population_by_gender_age.csv', testCoord)

	_, testPOIData, testCensusData = processData(dummyUsage, testPOIData, testCensusData)

	_, testFeatures = processDataForRegression(dummyUsage, testPOIData, testCensusData, filteredPOIs)

	testPred = LinearRegression.predict(testFeatures)
	print(testPred)

	statUsages = []
	for station in usageData: 
		statUsages.append(usageData[station])

	trainUseMed = np.median(statUsages)
	print(trainUseMed)

	print(testPred/trainUseMed)





	# plt.plot(us, p, 'ro')
	# plt.show()















