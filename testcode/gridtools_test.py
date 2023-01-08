from pathlib import Path

import gridtools as gt
import matplotlib.pyplot as plt
import numpy as np

dataroot = "../preprocessing_output/"
recs = gt.ListRecordings(dataroot, exclude=[])

recording = recs.recordings[0]
datapath = f"{dataroot}{recording}/"

grid = gt.GridTracks(datapath, finespec=False, verbose=True)

fig, ax = plt.subplots()
grid.plot_freq(ax)
plt.show()
