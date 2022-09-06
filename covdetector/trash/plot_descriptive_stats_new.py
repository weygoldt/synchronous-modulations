from enum import unique

import gridtools as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psutil import net_io_counters

from plotstyle import PlotStyle

s = PlotStyle()
dataroot = "../output/"
eventstats = pd.read_csv(dataroot + "eventstats.csv")
recs = gt.ListRecordings(dataroot)

# compute distance kde's
all_distances = []
dyad_distances = []
event_distances = []
n_all = 0
n_interact = 0
n_events = 0

for recording in recs.recordings:
    datapath = recs.dataroot + recording + "/"
    grid = gt.GridTracks(datapath, finespec=False)
    events = pd.read_csv(datapath + "events.csv")

    # get all distances
    n_all += len(grid.ids)
    dyad_ids = gt.utils.unique_combinations1d(grid.ids)
    for ids in dyad_ids:
        dyad = gt.Dyad(grid, ids)
        if dyad.overlap is True:
            all_distances.extend(dyad.dpos.tolist())

    # get unique interacting dyad distances
    dyads = []
    uniq1 = list(np.unique(events.id1))
    uniq2 = list(np.unique(events.id2))
    all_ids = np.unique(np.append(uniq1, uniq2))
    n_interact += len(all_ids)

    for idx in events.index:
        dy = sorted([events.id1[idx], events.id2[idx]])
        dyads.append(dy)
    unique_dyads = [list(x) for x in set(tuple(x) for x in dyads)]
    for ids in unique_dyads:
        dyad = gt.Dyad(grid, ids)
        if dyad.overlap is True:
            dyad_distances.extend(dyad.dpos.tolist())

    # get distances during events
    for idx in events.index:
        dyad = gt.Dyad(grid, [int(events.id1[idx]), int(events.id2[idx])])
        start = gt.utils.find_closest(dyad.times, float(events.start[idx]))
        stop = gt.utils.find_closest(dyad.times, float(events.stop[idx]))
        event_distances.extend(dyad.dpos[start:stop].tolist())
        n_events += 1

# convert to numpy arrays
all_distances = np.asarray(all_distances)
dyad_distances = np.asarray(dyad_distances)
event_distances = np.asarray(event_distances)

# make kdes for all
xlims_dpos = [0, 400]
bandwidth = 10

kde_all = gt.utils.kde1d(all_distances, bandwidth, xlims_dpos)
kde_interact = gt.utils.kde1d(dyad_distances, bandwidth, xlims_dpos)
kde_events = gt.utils.kde1d(event_distances, bandwidth, xlims_dpos)
