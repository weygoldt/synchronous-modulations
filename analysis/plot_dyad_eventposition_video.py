import os
from datetime import datetime

import gridtools as gt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from plotstyle import PlotStyle


def pathchecker(path):
    if path[-1] != "/":
        raise ValueError("ERROR: The path you specified misses a backslash at the end!")
        return None
    else:
        return path


def fancy_title(axis, title):
    if " " in title:
        split_title = title.split(" ", 1)
        axis.set_title(
            r"$\bf{{{}}}$".format(split_title[0]) + f" {split_title[1]}", pad=20
        )
    else:
        axis.set_title(r"$\bf{{{}}}$".format(title.replace(" ", r"\;")), pad=20)


s = PlotStyle()
trial = False  # enable to run in trial mode
darkmode = True  # disable to run in dark mode
fwindow = 5  # plotted frequency time window in mins
eventpad = 3  # time to plot before and after event in mins
maxmarkersize = 15  # size of fish markers
tail = 30  # tail length of markers

# set colors for dark or light mode
if darkmode:
    bgcolor = "#0d0d0d"
    framecolor = "#0d0d0d"
    brightcolor = "#cccccc"
    eventmarker = dict(color=s.colors["white"], alpha=0.05, lw=0)
    pathname = "eventvideo_dark_"
else:
    bgcolor = "#FFFFFF"
    framecolor = "#FFFFFF"
    brightcolor = "#000000"
    eventmarker = dict(color="lightblue", alpha=0.2, lw=0)
    pathname = "eventvideo_light_"

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# rcparams
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["axes.facecolor"] = bgcolor
plt.rcParams["figure.facecolor"] = bgcolor
plt.rcParams["figure.constrained_layout.use"] = False
mpl.rcParams["text.color"] = brightcolor
mpl.rcParams["axes.labelcolor"] = brightcolor
mpl.rcParams["xtick.color"] = brightcolor
mpl.rcParams["ytick.color"] = brightcolor

# select data
path = pathchecker("/home/weygoldt/Data/uni/efish/output/2016-04-20-18_49/")
events = pd.read_csv(path + "events.csv")
grid = gt.GridTracks(path, finespec=True, verbose=False)

# select events from event dataframe
selected_events = list(events.index)
if trial:  # select only first if in trial mode
    selected_events = selected_events[2:3]


# iterate over events
for i4 in selected_events:

    # make or get output directory for frames
    vidpath = gt.utils.simple_outputdir(path + f"{pathname}{i4}")

    # extract event from events dataframe
    event = events[events.index == i4]
    id1 = int(event.id1)
    id2 = int(event.id2)

    # initialize dyad of event ids
    dyad = gt.Dyad(grid, [id1, id2])

    # make axes limits for grid
    extent = np.array([[-10, 360], [-10, 360]])

    # create grid coordinates
    gridx = []
    gridy = []
    x_constructor = np.linspace(0, 350, 8)
    for x_coord in x_constructor:
        y_constructor = np.ones(8) * x_coord
        gridx.extend(x_constructor)
        gridy.extend(y_constructor)

    # set start and stop time
    if trial:
        start = gt.utils.find_closest(dyad.times, float(event.start))
    else:
        start = gt.utils.find_closest(dyad.times, float(event.start) - eventpad * 60)
    stop = gt.utils.find_closest(dyad.times, float(event.stop) + eventpad * 60)

    # get event indices on time vector of dyad
    eventstart = gt.utils.find_closest(dyad.times, float(event.start))
    eventstop = gt.utils.find_closest(dyad.times, float(event.stop))

    # make time index vector for iteration
    if trial:
        indices = np.arange(start, start + 1)
    else:
        indices = np.arange(start, stop)

    # get ylimit of freq plot
    ymin, ymax = gt.utils.get_ylims(
        dyad.fund_id1,  # fundamental frequencies of id1
        dyad.fund_id2,  # fundamental frequencies of id2
        dyad.times,  # shared times
        dyad.times[start],  # start timestamp
        dyad.times[stop],  # stop timestamp
        padding=0.2,  # padding factor
    )

    # get ylimit of freq plot time marker
    ymin_marker, ymax_marker = gt.utils.get_ylims(
        dyad.fund_id1,  # fundamental frequencies of id1
        dyad.fund_id2,  # fundamental frequencies of id2
        dyad.times,  # shared times
        dyad.times[start],  # start timestamp
        dyad.times[stop],  # stop timestamp
        padding=0.1,  # padding factor
    )

    # make arrays for modulation indicators: Linearly increasing vector from individuals
    # min to max EODf in plotted time range -  with same length of radius vector for plotted marker
    fmin1, fmax1 = np.min(dyad.fund_id1[start:stop]), np.max(dyad.fund_id1[start:stop])
    fmin2, fmax2 = np.min(dyad.fund_id2[start:stop]), np.max(dyad.fund_id2[start:stop])
    range1 = np.linspace(fmin1, fmax1, 100)
    range2 = np.linspace(fmin2, fmax2, 100)

    # radius vector for event marker size. For each event I the get the index of the closest value from the
    # two range vectors to the current EODf at a point in time and use the index to get
    # a radius from the radius vector.
    radius = np.linspace(maxmarkersize, 90, 100)

    # position styling
    markersizes = np.linspace(0, maxmarkersize, tail)  # decreasing marker size
    markeralphas = np.linspace(0, 1, tail)  #  decreasing marker alphas
    markeralphas_others = np.linspace(0, 1, tail)  # make other fish more transparent

    # plot positions
    i = 0  # cant remember what this was for
    # iterate over indices, one for each point on the time vector
    for i1 in tqdm(indices):

        # initialize plot
        hr = [1, 0.05, 1, 1]
        wr = [1, 0.05, 1, 1, 2]
        ny = len(hr)
        nx = len(wr)
        my_dpi = 96
        dpi_4k = 3840 / 13.3

        # 4k is 3840 × 2160 pixels. With the current size in inches,
        # we need the same size just with 3840/14 (long edge in pixes/long edge in inches) DPI.

        fig = plt.figure(figsize=(13.3, 7.48125), dpi=my_dpi)
        gs = GridSpec(ny, nx, figure=fig, height_ratios=hr, width_ratios=wr)
        gs.update(left=0.07, right=0.88, bottom=0.07, top=0.88, wspace=0.5, hspace=0.01)

        # add axes
        ax_grid = plt.subplot(gs[0:4, 0:4])
        ax_grid.set_aspect("equal")
        ax_freq = plt.subplot(gs[0, 4])
        ax_bar = plt.subplot(gs[1, 4], sharex=ax_freq)

        # plot grid electrodes
        ax_grid.scatter(gridx, gridy, **s.grid_electrodes)
        ax_grid.set_xlim(extent[0])
        ax_grid.set_ylim(extent[1])

        # get datetime for plot titles
        t = dyad.times[i1]  # current time
        index = np.arange(len(grid.times))  # index vector for grid times
        ti = index[grid.times == t][0]  # get idx for datetime from grid at current time
        datet = grid.datetimes[ti]  # get datetime at index from grid
        timestring = datet.strftime("%Y-%m-%d, %H:%M:%S")  # format to string

        # make axis titles
        padd = 20
        ax_grid.set_title("Position estimates", loc="center", pad=padd)  # timestamp
        ax_freq.set_title("EOD$f$", loc="center", pad=padd)

        # make tail position data arrays
        x1 = dyad.xpos_smth_id1[i1 - tail : i1]
        y1 = dyad.ypos_smth_id1[i1 - tail : i1]
        x2 = dyad.xpos_smth_id2[i1 - tail : i1]
        y2 = dyad.ypos_smth_id2[i1 - tail : i1]

        # make a tail
        for i2, (ms, ma) in enumerate(zip(markersizes, markeralphas)):
            ax_grid.plot(
                x1[i2], y1[i2], ".", **s.id1, markersize=ms, alpha=ma, label="id1"
            )
            ax_grid.plot(
                x2[i2], y2[i2], ".", **s.id2, markersize=ms, alpha=ma, label="id2"
            )

        # plot eod modulation indicator
        ms1 = radius[gt.utils.find_closest(range1, dyad.fund_id1[i1])]
        ms2 = radius[gt.utils.find_closest(range2, dyad.fund_id2[i1])]
        ax_grid.plot(x1[-1], y1[-1], ".", **s.id1, markersize=ms1, alpha=0.2)
        ax_grid.plot(x2[-1], y2[-1], ".", **s.id2, markersize=ms2, alpha=0.2)

        # iterate over all other fish in the grid instance
        gridevent = gt.utils.find_closest(grid.times, t)
        for track_id in grid.ids:
            gridt = grid.times[gridevent]

            if gridt in grid.times[grid.idx_v[grid.ident_v == track_id]]:

                # indices and data for fish
                indices = np.arange(
                    len(grid.times[grid.idx_v[grid.ident_v == track_id]])
                )
                tmpx = grid.xpos_smth[grid.ident_v == track_id]
                tmpy = grid.ypos_smth[grid.ident_v == track_id]

                # index for current time point
                i3 = indices[grid.times[grid.idx_v[grid.ident_v == track_id]] == gridt][
                    0
                ]

                # data subset with tail. If data not long enough for tail, use first point
                try:
                    tmpx_tail = tmpx[i3 - tail : i3]
                    tmpy_tail = tmpy[i3 - tail : i3]
                except IndexError:
                    tmpx_tail = tmpx[0:i3]
                    tmpy_tail = tmpy[0:i3]

                # plot current point with tail
                for i2, (ms, ma) in enumerate(zip(markersizes, markeralphas_others)):
                    ax_grid.plot(
                        tmpx_tail[i2],
                        tmpy_tail[i2],
                        ".",
                        color="grey",
                        markersize=ms,
                        alpha=ma,
                        label="other",
                    )

        # dynamically adjust xlims
        tstart, tstop = t - (fwindow * 60) * 0.2, t + (fwindow * 60) * 0.8
        istart, istop = gt.utils.find_closest(
            dyad.times, tstart
        ), gt.utils.find_closest(dyad.times, tstop)

        # plot EODf
        ax_freq.plot(
            dyad.times[istart:istop],
            dyad.fund_id1[istart:istop],
            **s.id1_slim,
            label="id1",
        )
        ax_freq.plot(
            dyad.times[istart:istop],
            dyad.fund_id2[istart:istop],
            **s.id2_slim,
            label="id2",
        )

        # # plot eventmarker on frequency axes, make smaller in each iteration
        # # to keep padding to y axis
        # eventmarker_start = dyad.times[eventstart]
        # eventmarker_stop = dyad.times[eventstop]
        # if dyad.times[eventstart] < dyad.times[istart]:
        #     eventmarker_start = dyad.times[istart]

        # # plot eventmaker only if it is still in the time range
        # if eventmarker_start < eventmarker_stop:
        #     ax_freq.axvspan(
        #         eventmarker_start,
        #         eventmarker_stop,
        #         **eventmarker,
        #         zorder=-1000,
        #     )

        # set axes limits
        ax_freq.set_ylim(ymin, ymax)
        ax_freq.set_xlim(tstart - 15, tstop)  # -15 for y axis padding

        # plot time marker
        line = mlines.Line2D(
            [t, t],
            [ymin_marker, ymax_marker],
            color=brightcolor,
            zorder=100,
            linewidth=1,
            alpha=1,
        )
        ax_freq.add_line(line)

        # add time scale indicator bar
        timescale = mlines.Line2D(
            [t + 20, t + 60 + 20],
            [0.5, 0.5],
            color=brightcolor,
            zorder=100,
            linewidth=2,
            alpha=1,
        )
        ax_bar.add_line(timescale)

        # get current EODf for legend entries
        f1 = dyad.fund_id1[i1]
        f2 = dyad.fund_id2[i1]

        # jerry-rig legend entries manually because number of other plotted fish changes
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=8,
                label="id1",
                markerfacecolor=s.id1_color,
                markeredgecolor=s.id1_color,
                color=bgcolor,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=8,
                label="id2",
                markerfacecolor=s.id2_color,
                markeredgecolor=s.id2_color,
                color=bgcolor,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                label="other",
                markersize=8,
                markerfacecolor="grey",
                markeredgecolor="grey",
                color=bgcolor,
            ),
        ]

        # add legend
        ax_freq.legend(
            handles=legend_elements,
            labels=[f"{f1:.2f} Hz", f"{f2:.2f} Hz", "Other fish"],
            # bbox_to_anchor=(-0.08, -1.8),
            bbox_to_anchor=(-0.11, -1.40),
            loc="lower left",
            ncol=1,
            labelspacing=1.2,
        )

        ax_grid.text(498, 221, "60 s", c=brightcolor, zorder=1000, size=SMALL_SIZE)
        ax_grid.text(432, 165, f"{timestring}", c=brightcolor, zorder=1000)
        ax_freq.set_ylabel("Hz", loc="top", rotation="horizontal", color=brightcolor)

        # remove ticks where not needed
        ax_grid.get_xaxis().set_ticks([])
        ax_grid.get_yaxis().set_ticks([])
        ax_freq.get_xaxis().set_ticks([])
        ax_bar.get_xaxis().set_ticks([])
        ax_bar.get_yaxis().set_ticks([])

        # remove spines where not needed
        ax_freq.spines.left.set_color(brightcolor)
        ax_freq.spines.right.set_color(framecolor)
        ax_freq.spines.top.set_color(framecolor)
        ax_freq.spines.bottom.set_color(framecolor)
        ax_grid.spines.left.set_color(framecolor)
        ax_grid.spines.right.set_color(framecolor)
        ax_grid.spines.top.set_color(framecolor)
        ax_grid.spines.bottom.set_color(framecolor)
        ax_bar.spines.left.set_color(framecolor)
        ax_bar.spines.right.set_color(framecolor)
        ax_bar.spines.top.set_color(framecolor)
        ax_bar.spines.bottom.set_color(framecolor)

        # show plot in trial, save if not trial
        if trial:
            plt.show()
        else:
            filename = str(i).rjust(5, "0")
            fig.savefig(vidpath + "/" f"{filename}.png", dpi=dpi_4k)
            i += 1
            plt.close()

    # convert to video if not trail after making all plots
    # samples at 3 Hz, coverted to 30 fps video -> time is x10 faster
    # if trial is False:
    #     os.system(
    #         f"cd {vidpath} && ffmpeg -r 30 -i %d.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -y -an ../eventvideo_{i4}.mp4"
    #     )

    # move back to code dir and repeat for nex event indcex
    # os.system("cd ~/Data/uni/efish/code/")
