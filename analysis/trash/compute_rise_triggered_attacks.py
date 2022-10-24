import gridtools as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Detect attacks, initiator and reactor of attacks, and then detect all rises for each of the two individuals of attack pairs.

# load data

datapath = "/home/weygoldt/Data/uni/efish/output/2016-04-16-18_45/"
grid = gt.GridTracks(datapath, finespec=False)

# attack detection

peakprom = 1  #
maxd = 25  # max interacting fish distance

dyad_ids = gt.utils.unique_combinations1d(grid.ids)

for ids in dyad_ids:
    dyad = gt.Dyad(grid, ids)

    if dyad.overlap is False:
        continue
