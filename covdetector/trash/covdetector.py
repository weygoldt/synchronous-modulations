import copy
import datetime
import glob
import os

import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import thunderfish as tf
from IPython import embed
from matplotlib.colorbar import Colorbar
from scipy.signal import find_peaks
from scipy.stats import mode
from tabulate import tabulate
from tqdm import tqdm

import functions as fs
import gridtools as gt
from plotstyle import PlotStyle
from termcolors import TermColor as tc


class SwxCov:
    def __init__(self, dyad, bin_width, maxlag, step, verbose=False):
        self.verbose = verbose
        self.covs, self.times, self.lags = self.sliding_window_xcov(
            dyad, bin_width, maxlag, step
        )

    def xcov(self, x, y, maxlag):
        """
        Cross covariance with lags.
        `x` and `y` must be one-dimensional numpy arrays with the same length.
        The return value has length 2*maxlag + 1.
        """

        # reduce maxlag size when data arrays are shorter than lags
        if maxlag >= len(x):
            raise Exception(
                tc.err("[ WARNING ]")
                + " Decrease xcorr maxlags or use longer data arrays! Data arrays are shorter than provided lags!"
            )
            return None

        # pad y vector according to maxlags
        py = np.pad(y.conj(), 2 * maxlag, mode="constant")

        # create all shifted y arrays (should be maxlags + 1 number of arrays)
        T = np.lib.stride_tricks.as_strided(
            py[2 * maxlag :],
            shape=(2 * maxlag + 1, len(y) + 2 * maxlag),
            strides=(-py.strides[0], py.strides[0]),
        )

        # pad x vector according to maxlags
        px = np.pad(x, maxlag, mode="constant")

        # calculate covariance
        covs = T.dot(px) / px.size - (T.mean(axis=1) * px.mean())

        # make lags vector to return
        lags = np.arange(-maxlag, maxlag + 1)

        return covs, lags

    def sliding_window_xcov(self, dyad, bin_width, maxlag, step=1):

        cov_radius = int((bin_width - 1) / 2)  # area left and right of center
        covs_times = []  # times at covs
        covs_m = []  # covariance matrix
        covs_lags = []  # covariance lags

        if self.verbose:
            print(
                f"{tc.GREEN}{tc.BOLD}[ SwxCov.sliding_window_xcov ]{tc.END} Computing rolling window cross covariance ..."
            )

        for idx in tqdm(range(cov_radius + 1, len(dyad.times) - cov_radius + 1, step)):
            bin = np.arange(idx - cov_radius - 1, idx + cov_radius)
            tp = dyad.times[idx]  # timepoint

            covs, lags = self.xcov(dyad.fund_id1[bin], dyad.fund_id2[bin], maxlag)

            covs_times = np.append(covs_times, tp)
            covs_m.append(np.array(covs))
            covs_lags = lags

        covs_m = np.array(covs_m)
        covs_m = covs_m.transpose()

        return covs_m, covs_times, covs_lags

    def maxima(self):
        """Returns maxima across one dimension of a 2d-array (e.g. an image).
        If given the 'x' argument to the dimesion, len(y) values are returned,
        each with the maximum of all values across the x axis for the respective index
        of the y axis point. In other words, the maximum values off datapoints across
        the x-axis. For time series data, where x is the time axis, the y argument results
        in the maxima across all y values.

        Args:
            image (2d array): 2d array, e.g. image, rolling window crosscorrelation, etc.
            lags (1d array): values for the axis of interest, e.g. time, crosscorrelation lags, etc.

        Returns:
            maxs (1d array): maxima across given dimension
            lags_maxs (1d array): values of the supplied lags values for the maxima
        """

        self.maxcovs = np.zeros(len(self.covs[0, :]))  # maximum covariances
        self.maxlags = np.zeros(len(self.covs[0, :]))  # lags at max cov

        for index in range(len(self.maxcovs)):
            self.maxcovs[index] = np.max(self.covs[:, index])
            self.maxlags[index] = self.lags[self.covs[:, index] == self.maxcovs[index]]

        return self.maxcovs, self.maxlags

    def minima(self):
        """Returns maxima across one dimension of a 2d-array (e.g. an image).
        If given the 'x' argument to the dimesion, len(y) values are returned,
        each with the maximum of all values across the x axis for the respective index
        of the y axis point. In other words, the maximum values off datapoints across
        the x-axis. For time series data, where x is the time axis, the y argument results
        in the maxima across all y values.

        Args:
            image (2d array): 2d array, e.g. image, rolling window crosscorrelation, etc.
            lags (1d array): values for the axis of interest, e.g. time, crosscorrelation lags, etc.

        Returns:
            maxs (1d array): maxima across given dimension
            lags_maxs (1d array): values of the supplied lags values for the maxima
        """

        self.mincovs = np.zeros(len(self.covs[0, :]))  # maximum covariances
        self.minlags = np.zeros(len(self.covs[0, :]))  # lags at max cov

        for index in range(len(self.mincovs)):
            self.mincovs[index] = np.min(self.covs[:, index])
            self.minlags[index] = self.lags[self.covs[:, index] == self.mincovs[index]]

        return self.mincovs, self.minlags


class CovDetector:
    def __init__(self, datapath, conf, ids=[], verbose=True, plot_process=True):

        # initialize class variables
        self.datapath = datapath

        bin_width = conf["swxcov_bin_duration"] * conf["track_rate"]
        self.bin_width = bin_width if fs.is_odd(bin_width) else (bin_width - 1)
        self.rate = conf["track_rate"]
        self.maxlag = conf["swxcov_lags"] * conf["track_rate"]
        self.step = conf["step"]
        self.radius = (self.bin_width - 1) / 2
        self.duration_thresh = conf["dyad_duration_threshold"]
        self.conf = conf
        self.plot_process = plot_process
        self.verbose = verbose

        # load data
        print(self.datapath)
        self.grid = gt.GridTracks(self.datapath, finespec=True)
        self.grid_plt = gt.GridTracks(self.datapath, finespec=False)
        self.grid_h = gt.GridTracks(self.datapath, finespec=False)
        self.grid_l = gt.GridTracks(self.datapath, finespec=False)

        # compute sex and eodf of individuals
        self.grid_plt.sex_ids()

        # apply bandpass filters
        self.grid_h.freq_bandpass(
            rate=conf["track_rate"],
            flow=conf["hcutoffs"][0],
            fhigh=conf["hcutoffs"][1],
            order=1,
            eod_shift=False,
        )
        self.grid_l.freq_bandpass(
            rate=conf["track_rate"],
            flow=conf["lcutoffs"][0],
            fhigh=conf["lcutoffs"][1],
            order=1,
            eod_shift=False,
        )

        # Bandpass filter the plot grid instance
        self.grid_plt.q10_norm()
        self.grid_plt.freq_bandpass(
            rate=3, flow=0.00004, fhigh=0.4, order=2, eod_shift=False
        )

        # extract ids if specified
        if len(ids) > 0:
            self.grid.extract_ids(ids)
            self.grid_h.extract_ids(ids)
            self.grid_l.extract_ids(ids)
            self.grid_plt.extract_ids(ids)
        else:
            ids = self.grid.ids

        # make unique id pairs
        self.id_dyads = fs.unique_combinations1d(ids)

        # initialize lists for event collection
        self.index_counter = 0  # counts the number of detected events
        self.event_start = []  # start timestamp of the event
        self.event_stop = []  # stop timestamp of the event
        self.event_id1 = []  # id 1 of interaction
        self.event_id2 = []  # id 2 of interaction
        self.restingeod_id1 = []  # restingeod of id 1
        self.restingeod_id2 = []  # restingeod of id 2
        self.initiator_covlags = []  # initiator determined by covariance lags
        self.approved = []  # whether event is visually approved or not

        self.id1_output = []
        self.id2_output = []
        self.id1_eodf = []
        self.id2_eodf = []
        self.id1_sex = []
        self.id2_sex = []
        self.time_start = []
        self.time_stop = []
        self.initiator = []

        for self.id_dyad in self.id_dyads:

            # detect events by sliding window cross covariance on two timescales
            self.detect_events(self.id_dyad)

            # skip if no overlap
            if self.overlap is False:
                continue

            # convert covariance indices to dyad and grid indices
            for event, peaks_l, peaks_h in zip(
                self.events_indices, self.events_peaks_l, self.events_peaks_h
            ):

                # convert indices
                self.find_event_indices(event)

                # pass temporary event peak indices to namespace to plot
                self.event_peak_h = peaks_h
                self.event_peak_l = peaks_l

                # verify plot appends 1 or 0 to self.approved and saves fig of event
                if self.plot_process:
                    self.split = True
                    while self.split:
                        self.verify_plot()
                        self.gui()

                # compute resting EODf of both IDs around event
                area_size = self.bin_width * 3
                before_event = []
                after_event = []
                if self.dyad_event[0] < area_size:
                    start = 0
                    stop = self.dyad_event[0]
                    before_event = [start, stop]
                else:
                    start = self.dyad_event[0] - area_size
                    stop = self.dyad_event[0]
                    before_event = [start, stop]

                if self.dyad_event[1] > (len(self.dyad.times) - 1) - area_size:
                    start = self.dyad_event[1]
                    stop = len(self.dyad.times) - 1
                    after_event = [start, stop]
                else:
                    start = self.dyad_event[1]
                    stop = self.dyad_event[1] + area_size
                    after_event = [start, stop]

                # for id1
                resting_area_id1 = np.append(
                    self.dyad.fund_id1[before_event[0] : before_event[1]],
                    self.dyad.fund_id1[after_event[0] : after_event[1]],
                )
                resting_id1 = mode(resting_area_id1)[0][0]

                # for id2
                resting_area_id2 = np.append(
                    self.dyad.fund_id2[before_event[0] : before_event[1]],
                    self.dyad.fund_id2[after_event[0] : after_event[1]],
                )
                resting_id2 = mode(resting_area_id2)[0][0]

                # append event indices and metadata to lists
                self.restingeod_id1.append(resting_id1)
                self.restingeod_id2.append(resting_id2)
                self.event_start.append(self.dyad_event[0])
                self.event_stop.append(self.dyad_event[1])
                self.index_counter += 1

            # add ids and event number to lists
            id1_array = np.ones(len(self.events_indices), dtype=int) * int(
                self.id_dyad[0]
            )
            id2_array = np.ones(len(self.events_indices), dtype=int) * int(
                self.id_dyad[1]
            )
            self.event_id1.extend(id1_array.tolist())
            self.event_id2.extend(id2_array.tolist())

    def detect_events(self, id_dyad):

        # initialiize dyad instances
        self.dyad = gt.Dyad(self.grid, id_dyad)
        self.dyad_h = gt.Dyad(self.grid_h, id_dyad)
        self.dyad_l = gt.Dyad(self.grid_l, id_dyad)
        self.dyad_plt = gt.Dyad(self.grid_plt, id_dyad)

        # load ids into class namespace
        self.id1 = int(self.dyad.id1)
        self.id2 = int(self.dyad.id2)

        # overlap control parameter
        self.overlap = True

        # check if dyad overlaps, exit function if not
        if self.dyad.overlap is False:
            print(
                tc.warn("[ CovDetector.detect_events ]")
                + f" Skipping dyad, {self.id1}, {self.id2} do not overlap."
            )
            self.overlap = False
            return None

        # compute overlap duration
        duration = np.max(self.dyad.times) - np.min(self.dyad.times)

        # check if duration is long enough, exit function if not
        if duration < self.duration_thresh:
            print(
                tc.warn("[ CovDetector.detect_events ]")
                + f" Skipping since overlapping duration of {self.id1}, {self.id2} not long enough!"
            )
            self.overlap = False
            return None
        else:
            print(
                tc.succ("[ CovDetector.detect_events ]")
                + f" {self.id1}, {self.id2} loaded."
            )

        # compute sliding window crosscovariances
        self.covs_h = SwxCov(
            dyad=self.dyad_h,
            bin_width=self.bin_width,
            maxlag=self.maxlag,
            step=self.step,
            verbose=self.verbose,
        )
        self.covs_l = SwxCov(
            dyad=self.dyad_l,
            bin_width=self.bin_width,
            maxlag=self.maxlag,
            step=self.step,
            verbose=self.verbose,
        )

        # check if match and make shared time and lags vectors
        if np.unique(self.covs_h.lags == self.covs_l.lags):
            self.covs_lags = self.covs_h.lags
        if np.unique(self.covs_h.times == self.covs_l.times):
            self.covs_times = self.covs_h.times

        # calculate max covs across time and respective lags at the max covs
        self.maxcovs_h, self.maxlags_h = self.covs_h.maxima()
        self.maxcovs_l, self.maxlags_l = self.covs_l.maxima()

        # make index vector for time vector
        int_times = np.arange(len(self.covs_times))

        # find peaks in maximum covs
        self.peaks_h, _ = find_peaks(
            self.maxcovs_h, prominence=self.conf["swxcov_peakprom_l"]
        )
        self.peaks_l, _ = find_peaks(
            self.maxcovs_l, prominence=self.conf["swxcov_peakprom_h"]
        )

        # find coocurring peaks and crop range
        self.events_indices, _, _ = fs.combine_cooccurences(
            self.peaks_l,
            self.peaks_h,
            int_times,
            self.radius,
            merge=False,
            crop=True,
        )

        # find coocurring peaks, cropped range might miss small peaks
        self.events_indices_uncropped, _, _ = fs.combine_cooccurences(
            self.peaks_l,
            self.peaks_h,
            int_times,
            self.radius,
            merge=False,
            crop=False,
        )

        # make boolean arrays for peaks
        peakbool_l = np.zeros(len(self.maxcovs_l), dtype=bool)
        peakbool_l[self.peaks_l] = 1
        peakbool_h = np.zeros(len(self.maxcovs_h), dtype=bool)
        peakbool_h[self.peaks_h] = 1

        # empty lists to collect peak positions
        self.events_peaks_l = []
        self.events_peaks_h = []

        for event_index in self.events_indices_uncropped:

            # make boolean array to indicate where event range is
            rangebool = np.zeros(len(self.maxcovs_l), dtype=bool)
            rangebool[event_index[0] : event_index[-1]] = 1

            # write bools to class data
            self.events_peaks_l.append(np.where(rangebool & peakbool_l)[0].tolist())
            self.events_peaks_h.append(np.where(rangebool & peakbool_h)[0].tolist())

            # get lags at peaks
            lags_l = self.maxlags_l[rangebool & peakbool_l]
            lags_h = self.maxlags_h[rangebool & peakbool_h]

            # compute mean of mean lags at peaks in event range
            mean_lag = np.mean([np.mean(lags_l), np.mean(lags_h)])

            # classify initiator based on mean
            if fs.get_sign(mean_lag) == 0:
                self.initiator_covlags.append(np.nan)
            elif fs.get_sign(mean_lag) < 0:
                self.initiator_covlags.append(self.id1)
            elif fs.get_sign(mean_lag) > 0:
                self.initiator_covlags.append(self.id2)
            else:
                print(
                    tc.err(
                        "Error getting initiator from covariance lags, 'mean_lag' is empty"
                    )
                )

    def find_event_indices(self, event):
        """Converts resulting event indices (based on the covariance time array) to the track dyad time array and finally to the time array of the whole dataset."""

        self.covs_event = [event[0], event[-1]]

        self.dyad_event = [
            np.where(self.dyad.times == self.covs_times[self.covs_event[0]])[0][0],
            np.where(self.dyad.times == self.covs_times[self.covs_event[1]])[0][0],
        ]

        self.data_event = [
            np.where(self.grid.times == self.dyad.times[self.dyad_event[0]])[0][0],
            np.where(self.grid.times == self.dyad.times[self.dyad_event[1]])[0][0],
        ]

    def verify_plot(self):
        def plot_spec(self, axis, style, finespec, xpadding, ypadding):
            # compute x and y limits
            xlower, xupper = (
                self.dyad.times[self.dyad_event[0] - xpadding[0]],
                self.dyad.times[self.dyad_event[1] + xpadding[1]],
            )

            ylower, yupper = freq_lim(
                self.dyad.fund_id1,
                self.dyad.fund_id2,
                self.dyad_event[0],
                self.dyad_event[1],
                ypadding,
            )

            f0, f1 = ylower, yupper  # frequency limitation
            t0, t1 = xlower, xupper  # time limitation

            if finespec == False:
                spec = axis.imshow(
                    tf.powerspectrum.decibel(self.grid.spec)[::-1],
                    extent=[
                        self.grid.times[0],
                        self.grid.times[-1] + (self.grid.times[1] - self.grid.times[0]),
                        0,
                        2000,
                    ],
                    vmin=-100,
                    vmax=-50,
                    **style.coarse_spec,
                )

            if finespec == True:

                # Why is the spec-track mismatch EXACTLY 5???

                f_mask = np.arange(len(self.grid.fill_freqs))[
                    (self.grid.fill_freqs >= f0 + 5) & (self.grid.fill_freqs <= f1 + 5)
                ]
                t_mask = np.arange(len(self.grid.fill_times))[
                    (self.grid.fill_times >= t0) & (self.grid.fill_times <= t1)
                ]

                axis.imshow(
                    tf.powerspectrum.decibel(
                        self.grid.fill_spec[
                            f_mask[0] : f_mask[-1], t_mask[0] : t_mask[-1]
                        ][::-1]
                    ),
                    extent=[t0, t1, f0, f1],
                    vmin=-100,
                    vmax=-50,
                    **style.fine_spec,
                )

            # plot frequency tracks
            axis.plot(
                self.dyad.times, self.dyad.fund_id1, **style.id1, label="track id1"
            )
            axis.plot(
                self.dyad.times, self.dyad.fund_id2, **style.id2, label="track id2"
            )

            # plot event marker
            eventline_x = self.dyad.times[self.dyad_event[0] : self.dyad_event[1]]
            eventline_y = np.ones(len(eventline_x)) * yupper - 7
            axis.plot(eventline_x, eventline_y, **style.event_markerbar)
            axis.scatter(
                eventline_x[0], eventline_y[0], marker="|", **style.event_markerbar
            )
            axis.scatter(
                eventline_x[-1],
                eventline_y[-1],
                marker="|",
                **style.event_markerbar,
            )

            # set axis limitations
            axis.set_ylim(ylower, yupper)
            axis.set_xlim(xlower, xupper)

            xlims = (xlower, xupper)
            rec_datetime = get_datetime(self.datapath)
            clock_time(xlims, rec_datetime, self.dyad.times, axis)

            # set y label

            # hide x ticks
            axis.set(xticklabels=[])  # Hide tick marks and spines

            # init legend
            leg = axis.legend(
                bbox_to_anchor=(1.03, 1.148, 0, 0),
                loc="upper right",
                ncol=2,
                frameon=False,
            )
            leg.get_texts()[0].set_text(f"ID 1 [{int(self.id1)}]")
            leg.get_texts()[1].set_text(f"ID 2 [{int(self.id2)}]")

        def plot_tracks(dyad, axis, style):
            axis.plot(dyad.times, dyad.fund_id1, **style.id1)
            axis.plot(dyad.times, dyad.fund_id2, **style.id2)

            xlims = (np.min(dyad.times), np.max(dyad.times))
            rec_datetime = get_datetime(self.datapath)
            clock_time(xlims, rec_datetime, dyad.times, axis)

            axis.set(xticklabels=[])
            axis.set_ylabel("fund. $f$  [Hz]")

        def plot_covs(self, covs, maxcovs, maxlags, axis, cbaxis, cbarlabel, style):

            # plot covariance heatmap
            heatmap = axis.imshow(
                covs,
                extent=[
                    self.covs_times[0],
                    self.covs_times[-1] + (self.covs_times[1] - self.covs_times[0]),
                    -self.maxlag,
                    self.maxlag,
                ],
                origin="lower",
                **style.cov_heatmap,
            )

            # add a center line
            axis.plot(
                np.arange(len(maxcovs)),
                np.zeros(len(maxcovs)),
                **style.center_line,
            )

            # multiply maxlags vby 0.99 to make visible in plot
            axis.plot(
                self.covs_times,
                maxlags * 0.95 / self.rate,
                **style.lags,
            )

            # add a colorbar to indicate covariance
            cbaxis = Colorbar(
                ax=cbaxis,
                mappable=heatmap,
                orientation="vertical",
                ticklocation="right",
            )

            # setup axes
            if cbarlabel:
                cbaxis.set_label(r"covariance", labelpad=10)

            axis.set_yticks([-self.maxlag, 0, self.maxlag])
            axis.set_ylabel("x-cov. lags [s]")

            xlims = (np.min(self.covs_times), np.max(self.covs_times))
            rec_datetime = get_datetime(self.datapath)
            clock_time(xlims, rec_datetime, self.covs_times, axis)

            axis.set(xticklabels=[])

        def plot_maxcovs(self, xpadding, axis, style):

            # make data to normalize in range for maxcovs plot
            maxcovs_l = self.maxcovs_l[
                self.covs_event[0] - xpadding[0] : self.covs_event[1] + xpadding[1]
            ]
            maxcovs_h = self.maxcovs_h[
                self.covs_event[0] - xpadding[0] : self.covs_event[1] + xpadding[1]
            ]
            times = self.covs_times[
                self.covs_event[0] - xpadding[0] : self.covs_event[1] + xpadding[1]
            ]

            # jerry rig something to be able to access event peak indices on normalized subset of the data
            peakbool_h = np.zeros(len(self.maxcovs_h), dtype=bool)
            peakbool_h[self.event_peak_h] = True
            peakbool_h_sub = peakbool_h[
                self.covs_event[0] - xpadding[0] : self.covs_event[1] + xpadding[1]
            ]

            peakbool_l = np.zeros(len(self.maxcovs_l), dtype=bool)
            peakbool_l[self.event_peak_l] = True
            peakbool_l_sub = peakbool_l[
                self.covs_event[0] - xpadding[0] : self.covs_event[1] + xpadding[1]
            ]

            # normalize both
            maxcovs_h_norm = maxcovs_h / np.max(maxcovs_h)
            maxcovs_l_norm = maxcovs_l / np.max(maxcovs_l)

            # set y lim based on normalized data
            ylower, yupper = freq_lim(
                maxcovs_l_norm, maxcovs_h_norm, 0, len(maxcovs_h), 0.2
            )

            axis.set_ylim(ylower, yupper)

            # plot normalized lines
            axis.plot(times, maxcovs_h_norm, **style.maxcovs_h)
            axis.plot(times, maxcovs_l_norm, **style.maxcovs_l)

            xlims = (np.min(times), np.max(times))
            rec_datetime = get_datetime(self.datapath)
            clock_time(xlims, rec_datetime, times, axis)

            # plot peaks
            axis.scatter(
                times[peakbool_h_sub],
                maxcovs_h_norm[peakbool_h_sub],
                **style.maxcovs_h_peaks,
            )
            axis.scatter(
                times[peakbool_l_sub],
                maxcovs_l_norm[peakbool_l_sub],
                **style.maxcovs_l_peaks,
            )

            axis.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=1,
            )
            axis.set_ylabel("norm. cov.")
            axis.set_xlabel("time [hh:mm]")

        def freq_lim(track1, track2, start, stop, padding):
            """Helper function to get frequency y axis limits from two fundamental frequency tracks.

            Args:
                track1 (array): First track
                track2 (array): Second track
                start (int): Index for first value to be plotted
                stop (int): Index for second value to be plotted
                padding (int): Padding for the upper and lower limit

            Returns:
                lower (float): lower limit
                upper (float): upper limit

            """
            allfunds_tmp = (
                np.concatenate(
                    [
                        track1[start:stop],
                        track2[start:stop],
                    ]
                )
                .ravel()
                .tolist()
            )
            lower = np.min(allfunds_tmp) - padding
            upper = np.max(allfunds_tmp) + padding
            return lower, upper

        def fancy_title(axis, title):
            split_title = title.split(" ", 1)
            axis.set(title=r"$\bf{{{}}}$".format(split_title[0]) + f" {split_title[1]}")

        def get_datetime(folder):
            print(folder)
            folder = folder[:-1]
            print(folder)
            rec_year, rec_month, rec_day, rec_time = os.path.split(
                os.path.split(folder)[-1]
            )[-1].split("-")
            rec_year = int(rec_year)
            rec_month = int(rec_month)
            rec_day = int(rec_day)
            try:
                rec_time = [
                    int(rec_time.split("_")[0]),
                    int(rec_time.split("_")[1]),
                    0,
                ]
            except:
                rec_time = [
                    int(rec_time.split(":")[0]),
                    int(rec_time.split(":")[1]),
                    0,
                ]

            rec_datetime = datetime.datetime(
                year=rec_year,
                month=rec_month,
                day=rec_day,
                hour=rec_time[0],
                minute=rec_time[1],
                second=rec_time[2],
            )

            return rec_datetime

        def clock_time(xlims, rec_datetime, times, axis):
            xlim = xlims
            dx = np.diff(xlim)[0]

            label_idx0 = 0
            if dx <= 20:
                res = 1
            elif dx > 20 and dx <= 120:
                res = 10
            elif dx > 120 and dx <= 1200:
                res = 60
            elif dx > 1200 and dx <= 3600:
                res = 300  # 10 min
            elif dx > 3600 and dx <= 7200:
                res = 1800  # 30 min
            else:
                res = 3600  # 60 min

            if dx > 1200:
                if rec_datetime.minute % int(res / 60) != 0:
                    dmin = int(res / 60) - rec_datetime.minute % int(res / 60)
                    label_idx0 = dmin * 60

            xtick = np.arange(label_idx0, times[-1], res)
            datetime_xlabels = list(
                map(lambda x: rec_datetime + datetime.timedelta(seconds=x), xtick)
            )

            if dx > 120:
                xlabels = list(
                    map(
                        lambda x: ("%2s:%2s" % (str(x.hour), str(x.minute))).replace(
                            " ", "0"
                        ),
                        datetime_xlabels,
                    )
                )
                rotation = 0
            else:
                xlabels = list(
                    map(
                        lambda x: (
                            "%2s:%2s:%2s" % (str(x.hour), str(x.minute), str(x.second))
                        ).replace(" ", "0"),
                        datetime_xlabels,
                    )
                )
                rotation = 45
            # ToDo: create mask
            mask = np.arange(len(xtick))[(xtick > xlim[0]) & (xtick < xlim[1])]
            axis.set_xticks(xtick[mask])
            axis.set_xticklabels(np.array(xlabels)[mask], rotation=rotation)
            axis.set_xlim(xlim)

        # load plotstyle namespace
        s = PlotStyle()

        # dynamically adjust plot window if event is close to edges
        if self.covs_event[0] < self.bin_width:
            xpadding_start = self.covs_event[0]
        else:
            xpadding_start = self.bin_width

        if len(self.covs_times) - self.covs_event[1] < self.bin_width:
            xpadding_stop = (len(self.covs_times) - 1) - self.covs_event[1]
        else:
            xpadding_stop = self.bin_width
        xpadding = [xpadding_start, xpadding_stop]

        # Grid setup
        hr = [1, 0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.5]
        wr = [1, 0.02]
        ny = len(hr)
        nx = len(wr)

        # init fig
        ypadding = 10
        cm = 1 / 2.54  # to convert inches to cm
        self.fig = plt.figure(figsize=(16 * cm, 24 * cm))

        # init gridspec
        gs = gridspec.GridSpec(
            ny, nx, figure=self.fig, height_ratios=hr, width_ratios=wr
        )
        gs.update(
            left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03
        )

        # make axes
        ax_spec = plt.subplot(gs[0, 0])  # spectrogram
        ax_bandpass_h = plt.subplot(gs[2, 0], sharex=ax_spec)  # higher bandpass funds
        ax_bandpass_l = plt.subplot(
            gs[3, 0], sharex=ax_bandpass_h
        )  # lower bandpass funds
        ax_covs_h = plt.subplot(
            gs[5, 0], sharex=ax_bandpass_l
        )  # higher bandpass covariances
        ax_covs_h_cb = plt.subplot(gs[5, 1])  # higher bandpass covariances colorbar
        ax_covs_l = plt.subplot(
            gs[6, 0], sharex=ax_covs_h
        )  # lower bandpass covariances
        ax_covs_l_cb = plt.subplot(gs[6, 1])  # lower bandpass covariances colorbar
        ax_thresh = plt.subplot(gs[8, 0], sharex=ax_covs_l)  # thresholding cov maxima

        # disable x labels for all
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        plt.setp(ax_bandpass_h.get_xticklabels(), visible=False)
        plt.setp(ax_bandpass_l.get_xticklabels(), visible=False)
        plt.setp(ax_covs_h.get_xticklabels(), visible=False)
        plt.setp(ax_covs_l.get_xticklabels(), visible=False)

        # plot spectrogram
        plot_spec(self, ax_spec, s, finespec=True, xpadding=xpadding, ypadding=ypadding)
        fancy_title(ax_spec, "A Dyad")

        # plot bandpass filtered tracks
        xlower, xupper = (
            self.dyad.times[self.dyad_event[0] - xpadding[0]],
            self.dyad.times[self.dyad_event[1] + xpadding[1]],
        )

        plot_tracks(self.dyad_h, ax_bandpass_h, s)
        fancy_title(
            ax_bandpass_h,
            "B Bandpass-filtered tracks",
        )
        ylower, yupper = freq_lim(
            self.dyad_h.fund_id1,
            self.dyad_h.fund_id2,
            self.dyad_event[0],
            self.dyad_event[1],
            1,
        )
        ax_bandpass_h.set_ylim(ylower, yupper)
        ax_bandpass_h.set_xlim(xlower, xupper)
        ax_bandpass_h.set_ylabel("filt. fund. $f$  [Hz]")

        plot_tracks(self.dyad_l, ax_bandpass_l, s)
        ylower, yupper = freq_lim(
            self.dyad_l.fund_id1,
            self.dyad_l.fund_id2,
            self.dyad_event[0],
            self.dyad_event[1],
            2,
        )
        ax_bandpass_l.set_ylim(ylower, yupper)
        ax_bandpass_l.set_xlim(xlower, xupper)
        ax_bandpass_l.set_ylabel("")

        # plot rwxcovs
        fancy_title(ax_covs_h, "C Sliding-window x-cov")

        # divide by 3 since covs are only computed once per second
        xlower, xupper = (
            self.covs_times[self.covs_event[0] - int(xpadding[0] / self.rate)],
            self.covs_times[self.covs_event[1] + int(xpadding[1] / self.rate)],
        )

        plot_covs(
            self,
            self.covs_h.covs,
            self.maxcovs_h,
            self.maxlags_h,
            ax_covs_h,
            ax_covs_h_cb,
            True,
            s,
        )
        ax_covs_h.set_xlim(xlower, xupper)

        plot_covs(
            self,
            self.covs_l.covs,
            self.maxcovs_l,
            self.maxlags_l,
            ax_covs_l,
            ax_covs_l_cb,
            False,
            s,
        )
        ax_covs_l.set_xlim(xlower, xupper)
        ax_covs_l.set_ylabel("")

        plot_maxcovs(self, xpadding, ax_thresh, s)

        ax_thresh.set_xlim(xlower, xupper)

        fancy_title(ax_thresh, "D Norm. cov. maxima")

        self.fig.align_labels()
        self.fig.savefig(
            self.datapath
            + f"covdetector_backend_dyad_{self.id_dyad[0]}_{self.id_dyad[1]}_index_{self.index_counter}.pdf"
        )
        plt.close()

    def gui(self):
        def freq_lim(track1, track2, start, stop, padding):
            """Helper function to get frequency y axis limits from two fundamental frequency tracks.

            Args:
                track1 (array): First track
                track2 (array): Second track
                start (int): Index for first value to be plotted
                stop (int): Index for second value to be plotted
                padding (int): Padding for the upper and lower limit

            Returns:
                lower (float): lower limit
                upper (float): upper limit

            """
            allfunds_tmp = (
                np.concatenate(
                    [
                        track1[start:stop],
                        track2[start:stop],
                    ]
                )
                .ravel()
                .tolist()
            )
            lower = np.min(allfunds_tmp) - padding
            upper = np.max(allfunds_tmp) + padding
            return lower, upper

        # dynamically adjust plot window if event is close to edges
        if self.covs_event[0] < self.bin_width:
            xpadding_start = self.covs_event[0]
        else:
            xpadding_start = self.bin_width

        if len(self.covs_times) - self.covs_event[1] < self.bin_width:
            xpadding_stop = (len(self.covs_times) - 1) - self.covs_event[1]
        else:
            xpadding_stop = self.bin_width

        xpadding = [xpadding_start, xpadding_stop]

        s = PlotStyle()

        self.fig, self.ax1 = plt.subplots(figsize=(298 * s.mm, 140 * s.mm))
        self.ax1.plot(self.dyad_plt.times, self.dyad_plt.fund_id1, **s.id1, label="id1")
        self.ax1.plot(self.dyad_plt.times, self.dyad_plt.fund_id2, **s.id2, label="id2")

        # plot bandpass filtered tracks
        xlower, xupper = (
            self.dyad_plt.times[self.dyad_event[0] - xpadding[0]],
            self.dyad_plt.times[self.dyad_event[1] + xpadding[1]],
        )

        ylower, yupper = freq_lim(
            self.dyad_plt.fund_id1,
            self.dyad_plt.fund_id2,
            self.dyad_event[0],
            self.dyad_event[1],
            1,
        )

        self.ax1.set_xlim(xlower, xupper)
        self.ax1.set_ylim(ylower, yupper)

        leg = self.ax1.legend(
            bbox_to_anchor=(1.03, 1.148, 0, 0),
            loc="upper right",
            ncol=2,
            frameon=False,
        )
        leg.get_texts()[0].set_text(f"ID 1 [{int(self.id1)}]")
        leg.get_texts()[1].set_text(f"ID 2 [{int(self.id2)}]")

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1930, 10, 1900, 520)

        self.plotoutput = {}
        print('Enter "?" for help!')
        self.weak_referece = self.fig.canvas.mpl_connect("key_press_event", self.press)
        plt.show()

        self.initiator.append(self.plotoutput["init_id"])
        self.approved.append(self.plotoutput["event"])
        self.time_start.append(self.plotoutput["lims"][0])
        self.time_stop.append(self.plotoutput["lims"][1])
        self.split = self.plotoutput["split"]
        self.id1_output.append(self.id1)
        self.id2_output.append(self.id2)
        self.id1_eodf.append(self.grid_plt.eodf[self.grid.ids == self.id1].tolist()[0])
        self.id2_eodf.append(self.grid_plt.eodf[self.grid.ids == self.id2].tolist()[0])
        self.id1_sex.append(self.grid_plt.sex[self.grid.ids == self.id1].tolist()[0])
        self.id2_sex.append(self.grid_plt.sex[self.grid.ids == self.id2].tolist()[0])

    def press(self, event):
        if event.key == "?":
            print("--------------------------------------------")
            print(
                "Enter "
                + tc.succ("[ 1/2 ]")
                + " to accept plotted region as event (1) or not (2"
            )
            print(
                "THEN enter "
                + tc.succ("[ 1/2/3 ]")
                + " to set the initiaor to ID1 (1), ID2 (2) or NAN (3)."
            )
            print(
                "THEN enter "
                + tc.succ("[ a ]")
                + " to accept the plotted x limits as event start and stop."
            )
            print(
                "THEN enter "
                + tc.succ("[ r/w ]")
                + " to reset variables (r) or save input (w) and close."
            )
            print("--------------------------------------------")

        elif len(self.plotoutput) == 0:
            if event.key == "1":
                self.plotoutput["event"] = 1
                print(f'Set event to {self.plotoutput["event"]}')
            elif event.key == "2":
                self.plotoutput["event"] = 0
                print(f'Set event to {self.plotoutput["event"]}')
            elif event.key == "r":
                print("variables reset!")
                self.plotoutput = {}
            elif event.key == "w":
                print(f"Finish input before saving. Current input: {self.plotoutput}")
            else:
                print("Invalid input, try again!")

        elif len(self.plotoutput) == 1:
            if event.key == "1":
                self.plotoutput["init_id"] = int(self.id1)
                print(f'Set initiator to {self.plotoutput["init_id"]}')
            elif event.key == "2":
                self.plotoutput["init_id"] = int(self.id2)
                print(f'Set initiator to {self.plotoutput["init_id"]}')
            elif event.key == "3":
                self.plotoutput["init_id"] = np.nan
                print(f'Set initiator to {self.plotoutput["init_id"]}')
            elif event.key == "r":
                print("variables reset!")
                self.plotoutput = {}
            elif event.key == "w":
                print(f"Finish input before saving. Current input: {self.plotoutput}")
            else:
                print("Invalid input, try again!")

        elif len(self.plotoutput) == 2:
            if event.key == "a":
                self.plotoutput["lims"] = self.ax1.get_xlim()
                print(f'Set limits to {self.plotoutput["lims"]}')
            elif event.key == "r":
                print("variables reset!")
                self.plotoutput = {}

        elif len(self.plotoutput) == 3:
            if event.key == "d":
                self.plotoutput["split"] = True
            if event.key == "w":
                self.plotoutput["split"] = False
                if self.plotoutput["event"] == 1:
                    self.fig.savefig(
                        self.datapath
                        + f"dyad_{self.id_dyad[0]}_{self.id_dyad[1]}_index_{self.index_counter}.pdf"
                    )
                plt.close()
            elif event.key == "r":
                print("variables reset!")
                self.plotoutput = {}

        elif len(self.plotoutput) == 4:
            if event.key == "w":
                plt.close()
            elif event.key == "r":
                print("variables reset!")
                self.plotoutput = {}

    def save(self):
        def interface():
            def promt():
                var = input("Save [s] or quit [q]? ")
                return var

            var = promt()
            save = False
            if var not in "sq":
                print("Invalid input, try again!")
                var = promt()
            elif var in "q":
                print(f"Ok bye!")
                quit()
            elif var in "s":
                save = True
                print(f"Saving ...")

            return save

        # write lists into pandas dataframe
        event_table = {
            "approved": self.approved,
            "id1": self.id1_output,
            "id2": self.id2_output,
            "eodf_id1": self.id1_eodf,
            "eodf_id2": self.id2_eodf,
            "sex_id1": self.id1_sex,
            "sex_id2": self.id2_sex,
            "start": self.time_start,
            "stop": self.time_stop,
            "initiator": self.initiator,
        }

        # export pandas dataframe to csv
        save = interface()
        try:
            if save:
                df = pd.DataFrame(event_table)
                df.to_csv(self.datapath + "events.csv", encoding="utf-8", index=False)
                print("\n" + tc.rainb("Yay! Data saved succesfully! Here's a preview:"))
                print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
        except ValueError:
            print(
                tc.err(
                    "Error making dataframe. Collected data arrays are not the same lengths."
                )
            )


def main(ids, reanalyze=False):
    """Main event detection loop. If reanalyze is True, only the IDs saved in the "events.csv" file that are marked as approved events will be randomly shuffled and reanalized to reduce time needed for analysis."""

    # create list of recordings in dataroot
    conf = fs.load_ymlconf("covdetector_conf.yml")
    recs = gt.ListRecordings(path=conf["data"], exclude=conf["exclude"])
    recording = recs.recordings[0]

    # create path to recording
    datapath = recs.dataroot + recording + "/"

    # if reanalyze only iterate through ids in events dataframe
    if reanalyze:
        df = pd.read_csv(datapath + "events.csv")
        ids = []
        for idx, id1, id2 in zip(df.index, df.id1, df.id2):
            if df.approved[idx] == 1:
                ids.append(id1)
                ids.append(id2)
        ids = np.unique(ids).tolist()

    # detect events using covariance events class
    events = CovDetector(datapath, conf=conf, ids=ids)
    events.save()


# ids to extract for trial runs
ids = np.array([], dtype=int)
ids = np.array([14150, 14151], dtype=int)

if __name__ == "__main__":
    main(ids, reanalyze=False)
