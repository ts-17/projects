""" data.py has key data utilities and preprocessing """
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from geopy.distance import geodesic
from config import ( BOTTLE_PATH, CAST_PATH, SHORE_PATH, 
    PREDICTOR_COLS, RESPONSE_COL, DATA_FRACTION, RANDOM_STATE ) 

def load_and_prep():
    # load bottle + cast data from csv files locally
    df_b = pd.read_csv(BOTTLE_PATH, low_memory=False)
    df_c = pd.read_csv(CAST_PATH,   low_memory=False)
    df = pd.merge(df_b,
                  df_c[['Cst_Cnt','Lat_Dec','Lon_Dec']],
                  on='Cst_Cnt', how='left')
    df = df.dropna(subset=['Lat_Dec','Lon_Dec'])
    # add dist to shore calculation
    shore = gpd.read_file(SHORE_PATH).cx[-130:-110,20:45]
    df = calculate_distance_to_shore(df, shore)
    # drop any rows missing predictors or response
    df = df.dropna(subset=PREDICTOR_COLS + [RESPONSE_COL])
    # subsample if needed (to use a fraction of the dataset..)
    if DATA_FRACTION < 1.0:
        df = df.sample(frac=DATA_FRACTION, random_state=RANDOM_STATE)
    return df

def calculate_distance_to_shore(df, shoreline):
    # flatten all shoreline coords
    pts = []
    for geom in shoreline.geometry:
        if geom.geom_type == 'LineString':
            pts.extend(geom.coords)
        else:
            for part in geom:
                pts.extend(part.coords)
    pts = np.array(pts)
    tree = cKDTree(pts)
    samples = df[['Lon_Dec','Lat_Dec']].to_numpy()
    _, idx = tree.query(samples, k=1)
    nearest = pts[idx]
    # geodesic in meters
    dists = [
        geodesic((lat,lon),(slat,slon)).meters
        for (lon,lat),(slon,slat) in zip(samples, nearest)
    ]
    df['d_from_shore_m'] = dists
    return df
