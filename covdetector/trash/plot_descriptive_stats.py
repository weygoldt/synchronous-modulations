import gridtools as gt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plottools.colors as c
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from plotstyle import PlotStyle


def set_boxplot_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


def distances(id_dyads, events, gridtracks):
    # extract pairwise distances for all dyads, interacting dyads and iteracting dyads during interaction.
    dpos_all = []
    dpos_interact = []
    dpos_events = []
    n_all = 0
    n_interact = 0

    # collect position and freq distance over all possible dyads
    ids = gt.utils.unique_combinations1d(gridtracks.ids)
    for id_dyad in tqdm(ids):
        dyad = gt.Dyad(gridtracks, id_dyad)
        if dyad.overlap is True:
            dpos_all.extend(dyad.dpos.tolist())
            n_all += 1

    # collect position and frequency distance over interacting dyads and during interactions.
    for id_dyad, idx in tqdm(zip(id_dyads, events.index)):
        dyad_indices = [events.dyad_start[idx], events.dyad_stop[idx]]
        dyad = gt.Dyad(gridtracks, id_dyad)
        dpos_interact.extend(dyad.dpos.tolist())
        dpos_events.extend(dyad.dpos.tolist()[dyad_indices[0] : dyad_indices[1]])
        n_interact += 1

    # convert to numpy arrays
    dpos_all = np.array(dpos_all)
    dpos_interact = np.array(dpos_interact)
    dpos_events = np.array(dpos_events)

    # compute kdes
    xlims_dpos = [0, 400]
    bandwidth = 10
    x_dpos_all, kde_dpos_all = gt.utils.kde1d(dpos_all, bandwidth, xlims_dpos)
    x_dpos_interact, kde_dpos_interact = gt.utils.kde1d(
        dpos_interact, bandwidth, xlims_dpos
    )
    x_dpos_event, kde_dpos_event = gt.utils.kde1d(dpos_events, bandwidth, xlims_dpos)

    return (
        dpos_all,
        dpos_interact,
        dpos_events,
        x_dpos_all,
        kde_dpos_all,
        x_dpos_interact,
        kde_dpos_interact,
        x_dpos_event,
        kde_dpos_event,
    )


def restingeods(id_dyads, events):
    df_resting = []
    n = 0
    for id_dyad, idx in zip(id_dyads, events.index):
        if np.isnan(events.initiator_approved[idx]):
            continue
        elif int(events.initiator_approved[idx]) == int(events.id1[idx]):
            initiator = events.restingeod_id1[idx]
            reactor = events.restingeod_id2[idx]
            n += 1
        elif int(events.initiator_approved[idx]) == int(events.id2[idx]):
            initiator = events.restingeod_id2[idx]
            reactor = events.restingeod_id1[idx]
            n += 1
        df_resting.append(initiator - reactor)

    # compute kde for init-react difference
    df_resting = np.array(df_resting)
    lims = [-82, 82]
    x_df, kde_df_resting = fs.kde1d(df_resting, 3, lims)

    return df_resting, x_df, kde_df_resting


def modulation_amps(id_dyads, events):
    amps_init = []
    amps_react = []
    for id_dyad, idx in zip(id_dyads, events.index):
        dyad = cs.Dyad(gridtracks, id_dyad)
        start = events.dyad_start[idx]
        stop = events.dyad_stop[idx]

        id1_max = np.max(dyad.fund_id1[start:stop])
        id1_basel = events.restingeod_id1[idx]
        amp1 = id1_max - id1_basel

        id2_max = np.max(dyad.fund_id2[start:stop])
        id2_basel = events.restingeod_id2[idx]
        amp2 = id2_max - id2_basel

        if np.isnan(events.initiator_approved[idx]):
            continue
        elif int(events.initiator_approved[idx]) == int(events.id1[idx]):
            amps_init.append(amp1)
            amps_react.append(amp2)
        elif int(events.initiator_approved[idx]) == int(events.id2[idx]):
            amps_init.append(amp2)
            amps_react.append(amp1)

    # compute kde for initiator and reactor amplitudes
    amps_init = np.array(amps_init)
    amps_react = np.array(amps_react)
    lims = fs.lims(amps_init, amps_react)
    xlims = [0, 45]
    x_amps_init, kde_amps_init = fs.kde1d(amps_init, 2, xlims)
    x_amps_react, kde_amps_react = fs.kde1d(amps_react, 2, xlims)

    return (
        amps_init,
        amps_react,
        x_amps_init,
        x_amps_react,
        kde_amps_init,
        kde_amps_react,
    )


def abs_df(id_dyads, events):
    df_abs_event = []
    df_abs_resting = []
    n = 0
    for id_dyad, idx in zip(id_dyads, events.index):
        if np.isnan(events.initiator_approved[idx]):
            continue
        else:
            dyad = cs.Dyad(gridtracks, id_dyad)
            start, stop = df.dyad_start[idx], df.dyad_stop[idx]
            res = abs(df.restingeod_id1[idx] - df.restingeod_id2[idx])
            eve = abs(
                np.median(dyad.fund_id1[start:stop])
                - np.median(dyad.fund_id2[start:stop])
            )
            df_abs_resting.append(res)
            df_abs_event.append(eve)

    # compute kdes for deltafs
    df_abs_resting = np.array(df_abs_resting)
    df_abs_event = np.array(df_abs_event)
    lims_df = fs.lims(df_abs_resting, df_abs_event)
    lims_df = [0, lims_df[1] + 5]
    x_df_resting, kde_absdf_resting = fs.kde1d(df_abs_resting, 2, lims_df)
    x_df_event, kde_absdf_event = fs.kde1d(df_abs_event, 2, lims_df)

    return (
        df_abs_event,
        df_abs_resting,
        x_df_event,
        x_df_resting,
        kde_absdf_event,
        kde_absdf_resting,
    )


def main():

    s = PlotStyle()
    colors = c.colors_muted
    base_color = colors["blue"]
    init_color = colors["orange"]
    react_color = colors["pink"]
    markeralpha = 0.4

    kdecolors = [colors["green"], colors["cyan"], colors["blue"]]
    pairedcolors = [colors["green"], colors["blue"]]

    # path and time setup
    recs = cs.ListRecordings(path=mt.PROCESSED_DATA_PATH, exclude=pp.EXCLUDE)
    datapath = recs.dataroot + recs.recordings[0] + "/"
    rec_datetime = fs.dir2datetime(datapath[:-1])

    # get data
    df = pd.read_csv(datapath + "events.csv")
    events = df[df.approved == 1]
    gridtracks = cs.GridTracks(datapath, finespec=True)

    # make id pairs
    id_dyads = [
        [int(events.id1[events.index == idx]), int(events.id2[events.index == idx])]
        for idx in events.index
    ]

    # compute distances
    (
        dpos_all,
        dpos_interact,
        dpos_events,
        x_dpos_all,
        kde_dpos_all,
        x_dpos_interact,
        kde_dpos_interact,
        x_dpos_event,
        kde_dpos_event,
    ) = distances(id_dyads, events, gridtracks)

    # collect resting eods of interacting dyads
    df_resting, x_df, kde_df_resting = restingeods(id_dyads, events)

    # collect modulation amplitudes for initiators and reactors
    (
        amps_init,
        amps_react,
        x_amps_init,
        x_amps_react,
        kde_amps_init,
        kde_amps_react,
    ) = modulation_amps(id_dyads, events)

    # compute nondirectional deltaf during and before events
    (
        df_abs_event,
        df_abs_resting,
        x_df_event,
        x_df_resting,
        kde_absdf_event,
        kde_absdf_resting,
    ) = abs_df(id_dyads, events)

    # Grid setup
    hr = [0.5, 0.3, 0.05, 1]
    wr = [1, 1]
    ny = len(hr)
    nx = len(wr)
    fig = plt.figure(figsize=(160 * s.mm, 210 * s.mm))

    # init gridspec
    grid = GridSpec(ny, nx, figure=fig, height_ratios=hr, width_ratios=wr)
    grid.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

    ax2 = fig.add_subplot(grid[0, 0:2])
    ax3 = fig.add_subplot(grid[1, 0:2])
    ax4 = fig.add_subplot(grid[3, 0])
    ax5 = fig.add_subplot(grid[3, 1])

    # distance kde
    ax2.fill_between(
        x_dpos_all, kde_dpos_all, color=kdecolors[0], lw=0, alpha=0.3, label="all dist."
    )
    ax2.fill_between(
        x_dpos_interact,
        kde_dpos_interact,
        color=kdecolors[1],
        lw=0,
        alpha=0.3,
        label="interact. dist.",
    )
    ax2.fill_between(
        x_dpos_event,
        kde_dpos_event,
        color=kdecolors[2],
        lw=0,
        alpha=0.3,
        label="event dist.",
    )
    ax2.legend()
    ax2.set_xlim(-10, None)
    ax2.set_ylim(-0.001, np.max(kde_dpos_event) + 0.001)

    # initiator vs reactor restingeod
    bp_width = 0.003  # width of boxplots
    bp_offset = 0.006  # offset for boxplots
    jitter_offset = 0.002  # offset for kdes
    jitter_width = 0.002  # with of scatter jitters

    ax3.plot(
        df_resting,
        np.full_like(df_resting, -jitter_offset),
        ".",
        color="grey",
        markeredgewidth=0.1,
        alpha=0.6,
    )
    bp1 = ax3.boxplot(
        df_resting,
        vert=0,
        positions=np.array([-bp_offset]),
        widths=np.array([bp_width]),
        manage_ticks=False,
    )
    set_box_color(bp1, "black")
    ax3.fill_between(x_df, kde_df_resting, alpha=0.3, color=pairedcolors[1], lw=0)
    ax3.axvline(0, linewidth=1, linestyle="dashed", color="grey", alpha=0.6)
    ax3.set_ylim(-0.006, np.max(kde_df_resting) + np.max(kde_df_resting) * 0.1)
    ax3.set_xlabel("resting EOD$f_{init} - $ resting EOD$f_{react}$  [Hz]")
    ax3.set_ylabel("PDF")
    loc = ticker.MultipleLocator(
        base=0.01
    )  # this locator puts ticks at regular intervals
    ax3.yaxis.set_major_locator(loc)
    ax3.set_ylim(-(bp_offset + bp_width * 1.5), None)

    # paired cloudplot with boxplots
    height1 = 0  # x where initiator points go
    height2 = 0.2  # x where reactor points go
    bp_width = 0.02  # width of boxplots
    bp_offset = 0.03  # offset for boxplots
    rc_offset = 0.05  # offset for kdes
    jitter_width = 0.005  # with of scatter jitters

    # plot scatterplot connectors
    for init, react in zip(amps_init, amps_react):
        y_tmp = [init, react]
        x_tmp = [height1, height2]
        ax4.plot(x_tmp, y_tmp, color="grey", linewidth=1, alpha=0.2)
    y_tmp = [np.mean(amps_init), np.mean(amps_react)]
    ax4.plot(x_tmp, y_tmp, color="black", linewidth=1.5, alpha=0.8)

    # plot scatterplot with jitter
    jit_x1 = np.random.normal(height1, jitter_width, size=len(amps_init))
    jit_x2 = np.random.normal(height2, jitter_width, size=len(amps_react))
    ax4.plot(
        jit_x1,
        amps_init,
        ".",
        alpha=0.6,
        markeredgewidth=0.1,
        color=pairedcolors[1],
        lw=0,
    )
    ax4.plot(
        jit_x2,
        amps_react,
        ".",
        alpha=0.6,
        markeredgewidth=0.1,
        color=pairedcolors[0],
        lw=0,
    )

    # plot boxplots
    bp1 = ax4.boxplot(
        amps_init,
        positions=np.array([height1 - bp_offset]),
        widths=np.array([bp_width]),
        manage_ticks=False,
    )
    set_box_color(bp1, "black")
    bp2 = ax4.boxplot(
        amps_react,
        positions=np.array([height2 + bp_offset]),
        widths=np.array([bp_width]),
        manage_ticks=False,
    )
    set_box_color(bp2, "black")

    # plot kde
    ax4.fill_betweenx(
        x_amps_init,
        np.full_like(kde_amps_init, height1 - rc_offset),
        kde_amps_init * (-1) + height1 - rc_offset,
        alpha=0.3,
        color=pairedcolors[1],
        lw=0,
    )

    # plot kde
    ax4.fill_betweenx(
        x_amps_react,
        np.full_like(kde_amps_react, height2 + rc_offset),
        kde_amps_react + height2 + rc_offset,
        alpha=0.3,
        color=pairedcolors[0],
        lw=0,
    )

    # plot df resting vs df during events

    # paired cloudplot with boxplots
    height1 = 0  # x where initiator points go
    height2 = 0.2  # x where reactor points go

    bp_width = 0.02
    bp_offset = 0.03  # offset for boxplots
    rc_offset = 0.05  # offset for kde
    jitter_width = 0.005

    # make connectors
    for res, eve in zip(df_abs_resting, df_abs_event):
        y_tmp = [res, eve]
        x_tmp = [height1, height2]
        ax5.plot(x_tmp, y_tmp, color="black", linewidth=1, alpha=0.2)
    y_tmp = [np.mean(df_abs_resting), np.mean(df_abs_event)]
    ax5.plot(x_tmp, y_tmp, color="black", linewidth=1.5, alpha=0.8)

    # make jitter, plot scatter
    jit_x1 = np.random.normal(height1, 0.008, size=len(df_abs_resting))
    jit_x2 = np.random.normal(height2, 0.008, size=len(df_abs_event))

    ax5.plot(
        jit_x1,
        df_abs_resting,
        ".",
        markeredgewidth=0.1,
        alpha=0.6,
        color=pairedcolors[1],
        lw=0,
    )
    ax5.plot(
        jit_x2,
        df_abs_event,
        ".",
        markeredgewidth=0.1,
        alpha=0.6,
        color=pairedcolors[0],
        lw=0,
    )

    # plot boxplots
    bp = ax5.boxplot(
        df_abs_resting,
        positions=np.array([height1 - 0.03]),
        widths=np.array([0.02]),
        manage_ticks=False,
    )
    set_box_color(bp, "black")

    bp = ax5.boxplot(
        df_abs_event,
        positions=np.array([height2 + 0.03]),
        widths=np.array([0.02]),
        manage_ticks=False,
    )
    set_box_color(bp, "black")

    # plot kdes
    ax5.fill_betweenx(
        x_df_resting,
        np.full_like(kde_absdf_resting, height1 - 0.05),
        kde_absdf_resting * (-1) + height1 - 0.05,
        alpha=0.3,
        color=pairedcolors[1],
        lw=0,
    )
    ax5.fill_betweenx(
        x_df_event,
        np.full_like(kde_absdf_event, height2 + 0.05),
        kde_absdf_event + height2 + 0.05,
        alpha=0.3,
        color=pairedcolors[0],
        lw=0,
    )
    ax5.set_ylim(-1, None)
    ax5.set_xlim(ax4.axes.get_xlim())

    s.fancy_title(ax2, "A")
    s.fancy_title(ax3, "B")
    s.fancy_title(ax4, "C")
    s.fancy_title(ax5, "D")

    ax2.set_ylabel("PDF")
    ax2.set_xlabel("fish distance [cm]")

    ax4.set_xlim(-0.15, 0.35)
    # specify x-axis locations
    x_ticks = [0, 0.2]
    # specify x-axis labels
    x_labels = ["init", "react"]
    # add x-axis values to plot
    ax4.set_xticks(ticks=x_ticks, labels=x_labels)
    ax4.set_ylabel("ind. event $\Delta f$ [Hz]")
    ax4.set_xlabel("interaction role")
    ylims = fs.lims(x_amps_init, x_amps_react)
    ax4.set_ylim(-1, None)

    # specify x-axis locations
    x_ticks = [0, 0.2]
    # specify x-axis labels
    x_labels = ["resting", "event"]
    # add x-axis values to plot
    ax5.set_xticks(ticks=x_ticks, labels=x_labels)
    ax5.set_ylabel("dyad $\Delta f$  [Hz]")
    ax5.set_xlabel("time measured")

    fig.align_labels()
    plt.savefig(gridtracks.datapath + "initiator_vs_reactors.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
