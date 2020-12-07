RAW_DIR = r"../data/raw/"
ALL_DIR = r"../data/all_data/"
PRE_DIR = r"../data/preprocessed_data/"
OUTPUTS = ["NO2", "NOX as NO2", "PM10", "PM2.5"]
# INPUTS = ["NO2", "NOX as NO2", "PM10", "PM2.5"]
METEO_FEATURES = {"Lufttemperatur" : 1, "Relativ Luftfuktighet" : 6, "Byvind" : 21, "Vindriktning": 3}

INPUTS = OUTPUTS + list(METEO_FEATURES.keys())

IN_STEPS = 48
OUT_STEPS = 24
STATIONS = [8779, 8780, 8781, 18644]
NUM_FEATURES = len(INPUTS)
NUM_OUTPUTS = len(OUTPUTS)
