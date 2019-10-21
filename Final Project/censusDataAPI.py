import urllib.request
import json

#Grabbing and parsing the JSON data
def census(lat,lng):
  #making the url
  MyUrl = ('https://geocoding.geo.census.gov/geocoder/geographies/coordinates'
           '?x=%s'
           '&y=%s'
           '&benchmark=Public_AR_Census2010&vintage=Census2010_Census2010&format=json') % (str(lng), str(lat))
  #grabbing the JSON result
  # print(MyUrl)
  response = urllib.request.urlopen(MyUrl)
  jsonRaw = response.read()
  jsonData = json.loads(jsonRaw)
  # print(jsonRaw)
  filteredData = jsonData['result']['geographies']['Census Tracts']
  # print(filteredData)
  if 'GEOID' in filteredData[0]:
    tractID = filteredData[0]['GEOID']
  else: 
    tractID = False

  return tractID

if __name__ == '__main__':

  lat = 42.328539
  longit = -71.100815

  la = 42.361780439606044
  lo= -71.10809952020645

  search = census(lat=lat,lng=longit)

  print(search)