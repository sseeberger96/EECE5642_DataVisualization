import urllib.request
import json

#Grabbing and parsing the JSON data
def GooglePlace(lat,lng,radius,types):
  #making the url
  AUTH_KEY = 'AIzaSyBldjyQpTGrMdrRVUEZinuoXHKX1BqLOD4'
  LOCATION = str(lat) + "," + str(lng)
  RADIUS = radius
  TYPES = types
  MyUrl = ('https://maps.googleapis.com/maps/api/place/nearbysearch/json'
           '?location=%s'
           '&radius=%s'
           '&types=%s'
           '&sensor=false&key=%s') % (LOCATION, RADIUS, TYPES, AUTH_KEY)
  #grabbing the JSON result
  # print(MyUrl)
  response = urllib.request.urlopen(MyUrl)
  jsonRaw = response.read()
  jsonData = json.loads(jsonRaw)
  return jsonData

#This is a helper to grab the Json data that I want in a list
def IterJson(place):
  x = [place['name'], place['reference'], place['geometry']['location']['lat'], 
         place['geometry']['location']['lng'], place['vicinity']]
  return x



if __name__ == '__main__':

  apiKey = 'AIzaSyBldjyQpTGrMdrRVUEZinuoXHKX1BqLOD4'

  lat = 42.328539
  longit = -71.100815

  searchType = 'school'

  radius = 500

  search = GooglePlace(lat=lat,lng=longit,radius=radius,types=searchType)

  results = []

  print(search['status'])

  # if search['status'] == 'OK':                                        
  for place in search['results']:
    # print(place['reference'])
    x = IterJson(place)
    results.append(x)
  
  print(results)
