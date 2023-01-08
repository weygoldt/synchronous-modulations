# [Synchronized EODf modulations](https://youtu.be/ihDTMcn7LWM) in the weakly electric fish *Apteronotus leptorynchus*

This protocol is a brief overview of the workflow of

  - [Reproducing the virtual environment](#reproducing-the-environment)
  - [Preprocessing](#preprocessing)
  - [Event detection](#event-detection)
  - [Data extraction](#data-extraction)


## Reproducing the virtual environment

To reproduce the virtual environment, install all nessecary Python 3.10.4 packages:

```sh
# clone repository
git clone https://github.com/weygoldt/syncmod-analysis.git

# move into repository
cd syncmod-analysis

# create virtual environment with pythons venv
venv create env

# install requirements
pip install -r requirements.txt

# download the wavetracker scripts for EOD tracking
git clone https://github.com/tillraab/wavetracker.git
```

## Preprocessing

The data used in this analysis was previously tracked using the [wavetracker](https://github.com/tillraab/wavetracker.git). In the current implementation, the algorithm is not accurate enough to handle complex frequency changes, especially when frequencies of two fish cross. Because of this, datasets must be imporoved by hand using the `EODsorter.py` gui tool provided by the wavetracker. After manually improving tracked frequencies, the following preprocessing steps are applied using the `gridtools` package created during this project. The preprocessing steps *can* include the following:

- Create a directory for the processed data
- Fill missing powers in the 'sign_v.npy' (on the original data!) that resulted by manually tracking with the gui and creates a backup of the old powers dataset `sign_v_backup.npy` in the data source directory.
- Load the hobologger file and extracts data for each seperate recording.
- Remove all unassigned frequencies from the dataset.
- Remove tracks below a certain duration threshold.
- Remove tracks below a certain tracking performance threshold.
- Compute positions of each tracked frequency by triangulation between the electrodes with the highest power for the respective frequency.
- Interpolate the full dataset (frequencies and positions, etc.).
- Normalize the frequency tracks by a Q10 value.
- Bandpass filter the frequency tracks to a certain target frequency.
- Smooth position estimates using a combination of velocity thresholding, median filtering and Savitzky-Golay smoothing.

Each individual preprocessing step, including its parameters, can be adjusted using a configuration file for the preprocessor. E.g. normalizing by Q10 might only be useful to estimate the sexes of the individuals. Bandpass filtering might only be useful in certain event detection scenarios. The configuration file is preconfigured to exclude theses steps and to perform a dry run to see if errors occur before committing to writing the changes to disk.

To access the configuration file, run the following in a terminal. The description and comments on each option should suffice to explain their function.

```sh
# defaults to visual studio code
datacleaner edit

# or specify a text editor
datacleaner edit --editor <e.g. vim, emacs, code, pycharm, ...>
```

Preprocessing parameters for the positon estimates can be explored using a [jupyter notebook](demos/position_preprocessing_explorer.ipynb) that visualizes changes for a single track.

After setting all paths and parameters the preprocessing can be executed by running `datacleaner run` in a terminal. If no errors are reported, set the option 'dry-run' to 'false' to write the preprocessed data to disk.

## Event detection

The [algorithm](/covdetector/) to detect synchronous modulation is based on a rolling window cross-covariance between every possible pair of frequency traces bandpass filtered to two timescales. In total, the frequency trace pair is filtered three times. Two narrow bandpass filters extract slow and fast modulations on the timescales relevant to synchronous modulations. Sliding window cross-covariances are computed for each pair of the two bandpass filtered track pairs. Only when the cross-covariances cross a threshold on both time scales and event is detected. For a visualization of the alhorithm, see chapter "Event detection" on the [conference poster](poster/main.pdf). A third, broader bandpass filter is applied remove the zero-frequency component, i.e. scale both tracks to zero for visual inspection on the matplotlib gui interface.

If the covariance threshold is crossed on both bandpass filter scales, a matplotlib interactive plot is opened presenting the detected event. A terminal interface now allows for manual validation and marking the onset and offset, as well as the initiator of the event. Events can also be connected if a single event is detected twice. After validating all detected events manually, the user is asked to confirm saving the output to disk.

All relevant settings of the event detector (e.g. bandpass cutoffs, cross covariance lags, etc.) can be adjusted in a [configuration file](/covdetector/covdetector_conf.yml).

## Data extraction & analysis

The fish positions on the grid where computed and processed using the coustom-written [gridtools](https://github.com/weygoldt/gridtools) package. Fish positions where estimated using triangulation between 6 electrodes of the highest power of the individuals EOD$f$. Position preprocessing included three filter steps. First, a velocity threshold was established, to reduce unrealistic jumps in position that would have required velocities that probably (not tested!) exceed the maximum speed of an individual. As a second step, a median filter was applied to remove jitter on a small spatial scale. Since this was not able to remove all sudden jumps, a Savitzky Golay filter was applied to the position tracks. The exploration of parameters that lead to the preprocessing steps used here can be explored using a [jupyter notebook](/demos/position_preprocessing_explorer.ipynb) written for this purpose. Position preprocessing, as well as other preprocessing parameters, such as interpolation of positions and tracked frequencies where conducted using the [datacleaner](https://github.com/weygoldt/gridtools/blob/master/gridtools/datacleaner_conf.yml) configuration file as part of the gridtools package.

Using the position tracks of individual fish I analyzed the movements of synchronously modulating pairs of fish further. Whether or not two individuals approach each other is determined by their relative velocities. Which of the two individuals approaches whom can be determined by the relative movement trajectory. If one fish aims towards the other while the distance decreases, this individual is the approaching one.

To be able to detect *physical interactions* in a broad sense as distinct events, I constructed a cost function that includes the relative velocities, heading trajectories and proximity of two synchronously modulating individuals. The cost function peaks where distances are small, fish approach each other fast, and if the heading is aimed towards each other. This cost function is now implemented as the method `gridtools.utils.find_interactions` in the gridtools package. The code leading to the cost function as well as the function incorporating it is illustrated in this [demo](/demos/relative_heading_angle_explorer.ipynb).

All code for further data extraction, e.g. of kernel density estimates, as well as plotting the resulting data is included in the [analysis](/analysis/) directory.
The resulting plots as well as a brief description is illustrated on the [poster](/poster/main.pdf) of this project.
