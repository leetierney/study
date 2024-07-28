import requests
import json

# Load JSON: json_data
with open("datasets/a_movie.json") as json_file:
    json_data = json.load(json_file)

def print_kvp(json_data):
    # Print each key-value pair in json_data
    for k in json_data.keys():
        print(k + ': ', json_data[k])

def print_title_year(json_data):
    print(json_data['Title'])
    print(json_data['Year'])

print_title_year(json_data)