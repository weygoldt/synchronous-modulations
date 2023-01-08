import datetime
from unicodedata import decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfiltfilt


def lowpass_filter(data, rate, cutoff, order=2):
    sos = butter(order, cutoff, btype="low", fs=rate, output="sos")
    y = sosfiltfilt(sos, data)
    return y


# path two both hobo logger files
path1 = "../output/logger1.csv"
path2 = "../output/logger2.csv"
logger1 = pd.read_csv(path1)
logger2 = pd.read_csv(path2)

# recode headings
dict = {"DateTime": "date", "Temp": "temp", "Lux": "lux"}
logger1.rename(columns=dict, inplace=True)
logger2.rename(columns=dict, inplace=True)

# convert to floats
for i, (temp, lux) in enumerate(zip(logger1["temp"], logger1["lux"])):
    temp_int, temp_dec = temp.split(" ")
    lux_int, lux_dec = lux.split(" ")
    float_temp = float(str(temp_int) + "." + str(temp_dec))

    if "." in lux:
        lux_int = int(str(lux_int).replace(".", ""))

    float_lux = float(str(lux_int) + "." + str(lux_dec))
    logger1["temp"][i] = float_temp
    logger1["lux"][i] = float_lux

for i, (temp, lux) in enumerate(zip(logger2["temp"], logger2["lux"])):
    temp_int, temp_dec = temp.split(" ")
    lux_int, lux_dec = lux.split(" ")
    float_temp = float(str(temp_int) + "." + str(temp_dec))

    if "." in lux:
        lux_int = int(str(lux_int).replace(".", ""))

    float_lux = float(str(lux_int) + "." + str(lux_dec))
    logger2["temp"][i] = float_temp
    logger2["lux"][i] = float_lux

# set index to datetime
format = "%d/%m/%y %H:%M:%S"
logger1["date"] = pd.to_datetime(logger1["date"], format=format)
logger1.set_index("date")
logger2["date"] = pd.to_datetime(logger2["date"], format=format)
logger1.set_index("date")

# concatenate
df = pd.concat([logger1, logger2], axis=0)
df = df.set_index("date")

# convert cols to numeric
for col in df:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# resample and interpolate to 1 Hz
df = df.resample("s").interpolate("linear")
df_new = df[(df.index < "2016-04-22 08:50:00") | (df.index > "2016-04-22 11:40:00")]
df = df_new.resample("s").interpolate("linear")

# lowpass filter
rate = 1
cutoff = 0.00006
df["temp_filt"] = lowpass_filter(df["temp"], rate, cutoff)
df["lux_filt"] = lowpass_filter(df["lux"], rate, cutoff)

# put negative values to 0
df["lux_filt"][df["lux_filt"] < 0] = 0

# plot to check
plt.plot(df.temp)
plt.plot(df.temp_filt)
plt.show()

plt.plot(df.lux_filt)
plt.show()

# save
df.to_csv("../efishdata-local/hobologger.csv")
