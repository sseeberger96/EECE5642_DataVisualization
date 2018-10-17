#!/user/bin/env python
import sys
import os
from os import listdir
from os.path import isfile, join

dataDictionary = {}

dataType = str(sys.argv[1])
root = str("./20news-bydate/20news-bydate-" + dataType)

folders = next(os.walk(root))[1]

for folder in folders:
	dataDictionary[folder] = {}
	path = root + "/" + folder + "/"
	files = [f for f in listdir(path) if isfile(join(path, f))]	

	for f in files:
		dataDictionary[folder][f] = ""

		with open(path + "/" + f, 'r', encoding = "ISO-8859-1") as myfile:
			dataDictionary[folder][f]=myfile.read().replace('\n', '')

print(dataDictionary)
	

