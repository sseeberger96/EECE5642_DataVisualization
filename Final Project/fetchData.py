import urllib.request
import json
import sys

html = urllib.request.urlopen("http://hubwaydatachallenge.org/api/v1/station/?format=json&username=tym0027&api_key=8b29d8b3bdcac800972839130daf4f9b55c0e2ae")

A = html.read().decode('utf-8')
B = json.loads(A)

''' # Meta data
print(B["meta"]["limit"])
print(B["meta"]["next"])
print(B["meta"]["offset"])
print(B["meta"]["previous"])
print(B["meta"]["total_count"])
'''

''' # Parse Data from hubway github and save to csv
csvFile = open("./stationData.txt", 'w')
csvFile.write("name,id,lat,lon")
for data in B["objects"]:
	s = str(data["name"]) +", " + str(data["terminalname"]) + ", " + str(data["point"]["coordinates"][1]) + ", " + str(data["point"]["coordinates"][0])
	csvFile.write(s + "\n")
	print(s + "\n")

csvFile.close()
'''

''' # Parse objects from state Data
for obj in B["objects"]:
	print(str("\n") + str(obj))
	input()
'''
		
