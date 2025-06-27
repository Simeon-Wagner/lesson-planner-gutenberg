import requests

url = 'https://npk-sensor-facitec-2025-140378133387.southamerica-east1.run.app/data'
myobj = {'Greet': 'Someone'}

x = requests.post(url, json = myobj)

print(x.text)
