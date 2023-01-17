import matplotlib.pyplot as plt
import modules.functions as fs
import numpy as np
from modules.plotstyle import PlotStyle

import gridtools as gts


def main(recs, conf):
    """Executes all preprocessing steps into grid gt across all recordings in
    the specified directory. Data is saved to subfolders into the specified
    outout directory. Preprocessing parameters can be tweaked in the "settings.py"
    module.
    """
    s = PlotStyle()

    for recording in recs.recordings:

        # create path to recroding
        datapath = recs.dataroot + recording + "/"

        # set output dir name same as original dirname
        output_path = conf["output"] + recording

        # create output dir
        fs.simple_outputdir(conf["output"])
        fs.simple_outputdir(output_path)

        # load gt into gt object
        gt = gts.GridTracks(datapath, finespec=conf["fine_spec"])

        # compute powers for newly added tracks using wavetracker gui
        if conf["fill_powers"]:
            gt.fill_powers()

        if conf["load_logger"]:
            gt.load_logger()

        # remove nans in dataset (unassigned frequencies)
        if conf["remove_nans"]:
            gt.remove_nans()

        # remove short tracks
        if conf["remove_short"]:
            dur_thresh = conf["duration_threshold"]
            gt.remove_short(dur_thresh)

        # remove poorly tracked tracks
        if conf["remove_poor"]:
            perf_thresh = conf["performance_threshold"]
            gt.remove_poor(perf_thresh)

        # triangulate positions of tracked ids on the grid
        if conf["compute_positions"]:
            num_el = conf["electrodes"]
            gt.positions(num_el)

        # interpolate positions  estimations and frequency tracks
        if conf["interpolate"]:
            gt.interpolate()

        if conf["norm_q10"]:
            gt.load_logger()
            gt.q10_norm(
                normtemp=conf["tempnorm"]["normtemp"], q10=conf["tempnorm"]["norm_q10"]
            )

        if conf["bandpass"]:
            gt.freq_bandpass(
                rate=conf["freq_bandpass"]["rate"],
                flow=conf["freq_bandpass"]["flow"],
                fhigh=conf["freq_bandpass"]["fhigh"],
                order=conf["freq_bandpass"]["order"],
                eod_shift=conf["freq_bandpass"]["eod_shift"],
            )

        # smooth position estimations
        if conf["smooth_positions"]:
            params = conf["position_processing"]
            gt.smooth_positions(params)

        # plot a preview of the dataset
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        gt.plot_spec(axs[0])
        gt.plot_pos(axs[1], legend=False)
        xlims = [np.min(gt.times), np.max(gt.times)]
        s.clocktime(datapath, axs[0], gt.times, xlims)
        plt.show()

        # save to file
        if conf["dry_run"] is False:
            gt.save(output_path, check=True)


if __name__ == "__main__":

    # open config file
    conf = fs.load_ymlconf("data/datacleaner_conf.yml")

    # get paths from config
    dataroot = conf["data"]
    exclude = conf["exclude"]

    # get list of recordings in dataroot
    recs = gts.ListRecordings(path=dataroot, exclude=exclude)

    # if recordings are selected in conf, only use those
    if len(conf["include_only"]) > 0:
        recs.recordings = conf["include_only"]

    # run preprocessing on list of recordings
    main(recs, conf)
