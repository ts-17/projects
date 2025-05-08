""" plot_eda.py is a simple script to perform some basic EDA and plotting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import geopandas as gpd
from scipy.spatial import cKDTree
from geopy.distance import geodesic
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# physical / chemical parameters
PREDICTOR_COLS = [
    'Depthm', 'Salnty', 'O2ml_L', 'STheta', 'O2Sat', 'Oxy_mumol/Kg',
    'ChlorA', 'Phaeop', 'PO4uM', 'SiO3uM', 'NO2uM', 'NO3uM', 'NH3uM',
    'Lat_Dec', 'Lon_Dec', 'd_from_shore_m',
]
RESPONSE_COL = 'T_degC'
TEST_SIZE    = 0.20

df_bottle = pd.read_csv('ocean_modeling/calcofi_data/194903-202105_Bottle.csv', low_memory=False)
df_cast   = pd.read_csv('ocean_modeling/calcofi_data/194903-202105_Cast.csv',   low_memory=False)

# merge lat/lon from cast onto bottle
df = pd.merge(
    df_bottle,
    df_cast[['Cst_Cnt','Lat_Dec','Lon_Dec']],
    on='Cst_Cnt',
    how='left'
).dropna(subset=['Lat_Dec','Lon_Dec'])

# calculate distance to shoreline
shoreline = gpd.read_file(
    'ocean_modeling/calcofi_data/coastlines_10m/ne_10m_coastline/ne_10m_coastline.shp'
).cx[-140:-100,10:50]

def calculate_distance_to_shore(df, shoreline_gdf):
    pts = []
    for geom in shoreline_gdf.geometry:
        if geom.geom_type == 'LineString':
            pts.extend(geom.coords)
        else: 
            for part in geom:
                pts.extend(part.coords)
    pts = np.array(pts)
    tree = cKDTree(pts)
    sample_pts = df[['Lon_Dec','Lat_Dec']].to_numpy()
    d, idx = tree.query(sample_pts, k=1)
    nearest = pts[idx]
    meters = [
        geodesic((lat,lon),(slat,slon)).meters
        for (lon,lat),(slon,slat) in zip(sample_pts, nearest)
    ]
    out = df.copy()
    out['d_from_shore_m'] = meters
    return out

regression_df = calculate_distance_to_shore(df, shoreline)
# drop missing rows (not needed for eda plots)
# required = PREDICTOR_COLS + [RESPONSE_COL]
# regression_df = regression_df.dropna(subset=required)

# hexbin of temperature
# heatmap on map with shoreline overlay
fig, ax = plt.subplots(figsize=(8, 6))

# hexbin of temperature
hb2 = ax.hexbin(
    regression_df['Lon_Dec'], regression_df['Lat_Dec'],
    C=regression_df[RESPONSE_COL],
    reduce_C_function=np.mean,
    gridsize=200, mincnt=1, cmap='plasma'
)

# overlay shoreline
shoreline.plot(ax=ax, edgecolor='black', linewidth=0.5)

# plot labels and colorbar
fig.colorbar(hb2, ax=ax).set_label(f'Mean {RESPONSE_COL}')
ax.set(xlabel='Longitude', ylabel='Latitude', title=f'CalCOFI Samples with the California Shoreline')
ax.set_ylim([15,50])
ax.set_xlim([-165,-105])
plt.tight_layout()
plt.show()

# commented for speed, these are functional:
#### scatter matrix
# sm_cols = PREDICTOR_COLS + [RESPONSE_COL]
# pd.plotting.scatter_matrix(
#     regression_df[sm_cols],
#     diagonal='kde', figsize=(12,12), alpha=0.5
# )
# plt.suptitle("Scatter Matrix of Predictors & Response", y=1.02)
# plt.tight_layout()
# plt.show()

#### correlation matrix
# corr = regression_df[sm_cols].corr()
# fig, ax = plt.subplots(figsize=(10,8))
# sns.heatmap(
#     corr, annot=True, fmt=".2f",
#     cmap='coolwarm', square=True, linewidths=0.5, ax=ax
# )
# ax.set_title("Correlation Matrix")
# plt.tight_layout()
# plt.show()