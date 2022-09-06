# Synchronized EODf modulations in the weakly electric fish *Apteronotus leptorynchus*

This protocol is a brief overview of the workflow of 

  - [Reproducing the environment](#reproducing-the-environment)
  - [Preprocessing](#preprocessing)
  - [Event detection](#event-detection)
  - [Data extraction](#data-extraction)
  - [Plotting results](#plotting-results)


## Reproducing the environment

To reproduce the environment, install all nessecary Python 3.10.4 packages:

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

## Data extraction

## Plotting results