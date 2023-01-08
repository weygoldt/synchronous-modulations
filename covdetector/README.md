# Covdetector

Detect synchronously modulating individuals on grid recordings.

## Initialization

The Covdetector runs on preprocessed (tracked and interpolated) electrode grid
recordings in the wavetracker output format (directory of numpy files).
It uses the rolling covariance between two tracks fitered to two different time
scales. To use it, add the path to the dataset and a path to an existing output
directory to the cofiguration file.

## Detecting events

To run the Covdetector, activate the virtual environment containing all required
packages, navigate into the directory containing the covdetector script and the
config-file and execute it.

```bash
source .env/bin/activate # activate venv
cd covdetector
python covdetector.py
```

If an event (covariance threshold crossed on both time scales in same window)
is detected, a simple matplotib gui opens.
The gui is used to verify the event, mark on- and offset using the zoom
functions and to select the initiator, i.e. the one that rose its frequency
first. If a gui window opens, the terminal provides a guideline on how to use it.
The script execution can be paused between recordings.
In this case, a .yml file is created that the script loads at startup to
continue where you left off. When the script finishes, a csv file will be created
containing data for all verified events.


