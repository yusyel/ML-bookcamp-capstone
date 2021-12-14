import requests


url = 'http://localhost:9696/predict'




test = {
	"vehicle_class": "subcompact",
	"engine_size": 1.5,
	"cylinders": 3,
	"fuel_type": "z",
	"fuel_consumption_city":8.3,
	"fuel_consumption_hwy":6.4,
	"fuel_consumption_comb_mpg":38


}
response = requests.post(url, json=test).json()
print(response)