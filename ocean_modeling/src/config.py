""" config.py contains the critical options for the dataset and filepaths.. """
PREDICTOR_COLS = [
    'Depthm','Salnty','O2ml_L','STheta','O2Sat','Oxy_mumol/Kg',
    'ChlorA','Phaeop','NO2uM','NO3uM','SiO3uM','T_degC', 'PO4uM',
    'Lat_Dec','Lon_Dec','d_from_shore_m'
]
RESPONSE_COL         = 'NH3uM'
DATA_FRACTION        = 1.0
TEST_SIZE            = 0.2
RANDOM_STATE         = 9382357
RUN_FEATURE_SELECTION= True

# raw data locations
BOTTLE_PATH = 'ocean_modeling/calcofi_data/194903-202105_Bottle.csv'
CAST_PATH   = 'ocean_modeling/calcofi_data/194903-202105_Cast.csv'
SHORE_PATH  = 'ocean_modeling/calcofi_data/coastlines_10m/ne_10m_coastline/ne_10m_coastline.shp'
