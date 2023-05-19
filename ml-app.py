# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
import joblib

app = Flask(__name__)

# Load data and models outside of route handler functions
cities = pd.read_csv('data/cities-clean.csv')
sncf = pd.read_csv('data/sncf-geocoordinates.csv')
ratp = pd.read_csv('data/ratp-geocoordinates.csv')
velib_geo = pd.read_csv("data/velib-geo.csv")
new_stations = pd.read_csv('data/new-stations.csv')
loaded_model = joblib.load('finalized_model.sav')
sncf_coords = np.array(sncf[['latitude', 'longitude']])
ratp_coords = np.array(ratp[['stop_lat', 'stop_lon']])
velib_coords = np.array(velib_geo[['lat', 'lon']])
new_coords = np.array(new_stations[['latitude', 'longitude']])


# Your helper functions go here
# Helper functions
def get_citycode(city):
    df = cities[cities.commune == city]
    return df.reset_index().code_insee[0]

def get_geocords(address, insee):
    address = str(address)
    insee = str(insee)
    
    params = (
        ('q', address),
        ('citycode', insee)
    )

    response = requests.get('https://api-adresse.data.gouv.fr/search/', params=params)

    longitude, latitude = response.json()['features'][0]['geometry']['coordinates']

    return longitude, latitude

def extract_query(address, city):
    insee = get_citycode(city)
    longitude, latitude = get_geocords(address, insee)
    return insee, longitude, latitude

def spherical_dist(pos1, pos2, r=6731):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def retrieve_info(array):
    a3 = list(filter(lambda x: x <= 3, array))
    a2 = list(filter(lambda x: x <= 2, a3))
    a1 = list(filter(lambda x: x <= 1, a2))

    n3 = len(a3)
    n2 = len(a2)
    n1 = len(a1)

    dmin = min(array)

    return n1, n2, n3, dmin

def get_dist_infos(lat, lon):
    sncf_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), sncf_coords)
    )

    n1_sncf, n2_sncf, n3_sncf, dist_sncf = retrieve_info(sncf_array)

    ratp_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), ratp_coords)
    )

    n1_ratp, n2_ratp, n3_ratp, dist_ratp = retrieve_info(ratp_array)

    new_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), new_coords)
    )

    n1_new, n2_new, n3_new, dist_new = retrieve_info(new_array)

    velib_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), velib_coords)
    )

    n1_velib, n2_velib, n3_velib, dist_velib = retrieve_info(velib_array)

    gare_proche = min(dist_sncf, dist_ratp)
    gare_proche_sq = gare_proche ** 2

    return gare_proche, gare_proche_sq, n3_sncf, dist_new, n3_velib

def get_all_infos(adresse, ville, local):
    insee, lon, lat = extract_query(adresse, ville)

    gare_proche, gare_proche_sq, n3_sncf, dist_new, n3_velib = get_dist_infos(lat, lon)

    year = '2020'

    code_departement = str(insee)[:2]

    df = pd.DataFrame({
        'gare_proche': gare_proche,
        'gare_proche_sq': gare_proche_sq,
        'n3_sncf': n3_sncf,
        'dist_new': dist_new,
        'n3_velib': n3_velib,
        'year': year,
        'code_insee': insee,
        'type_local': local,
        'code_departement': code_departement
    }, index=[0])

    num_cols = [
        'gare_proche',
        'gare_proche_sq',
        'n3_sncf',
        'dist_new',
        'n3_velib',
    ]

    cat_cols = ['year', 'type_local', 'code_insee', 'code_departement']

    relevant_cols = [
        y for x in [x for x in 
            [num_cols, cat_cols]
            ] for y in x
    ]

    df = df.loc[:, relevant_cols]

    return df



# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    address = data['address']
    city = data['city']
    location_type = data['type']
    
    # get all info for the provided address, city and location type
    df = get_all_infos(address, city, location_type)
    
    # predict
    prediction = loaded_model.predict(df)
    
    # output prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
