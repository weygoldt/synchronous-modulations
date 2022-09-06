import gridtools as gt
import matplotlib.pyplot as plt
import numpy as np

datapath = "../output/2016-04-09-22_25/"
grid = gt.GridTracks(datapath)

# look at all frequency tracks on a spectrogram
fig, ax = plt.subplots()
grid.plot_spec(ax)
plt.show()

# look at all position estimates
fig, ax = plt.subplots()
grid.plot_pos(ax)
plt.show()

# extract data of a single interesting individual from the previous plot
fish = gt.Monad(grid, 26787)

# plot position and frequency of individual
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
iois = np.arange(30000, 32000)  # indices of interest
ax[0].plot(fish.times[iois], fish.fund[iois])
ax[1].plot(fish.xpos_smth[iois], fish.ypos_smth[iois])
ax[1].set_xlim(0, 350)
ax[1].set_ylim(0, 350)
plt.show()

# extract data of two individuals
ids = [26789, 26788]
dyad = gt.Dyad(grid, ids)

# extract some interesting time points
start = gt.utils.find_closest(dyad.times, 26500)
stop = gt.utils.find_closest(dyad.times, 27500)
iois = np.arange(start, stop)

# plot fundamental frequencies and smoothed positions
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(dyad.times[iois], dyad.fund_id1[iois])
ax[0].plot(dyad.times[iois], dyad.fund_id2[iois])
ax[1].plot(dyad.xpos_smth_id1[iois], dyad.ypos_smth_id1[iois])
ax[1].plot(dyad.xpos_smth_id2[iois], dyad.ypos_smth_id2[iois])
plt.show()

# extract n interacting individuals
ids = ids = [26789, 26788, 26368]
col = gt.Collective(grid, ids)

# extract some interesting time points
start = gt.utils.find_closest(col.fish[0].times, 26500)
stop = gt.utils.find_closest(col.fish[0].times, 27500)
iois = np.arange(start, stop)

# plot
fig, ax = plt.subplots()
col.plot_pos(ax)
plt.show()

# look at some specific point in time
fig, ax = plt.subplots()
for fish in col.fish:
    ax.plot(fish.xpos_smth[iois], fish.ypos_smth[iois])
plt.show()
