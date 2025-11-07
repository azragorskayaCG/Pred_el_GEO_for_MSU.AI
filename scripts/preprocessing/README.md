This folder contains scripts for data preprocessing.

## `Preprocessing.py`

The `Preprocessing` class has functions for preprocessing all files within a directory:

- `mini_files_to_big`: combine all files of the directory into a single dataset
- `change_resolution`: resample the dataset to a new time resolution: convert hourly data to daily resolution
- `fill_missing_by_interpolation`: fill missing values using linear interpolation when missing segment is shorter than 3 days 
- `fill_missing_by_ML`: fill missing values using InterativeImputer when missing segments are longer than 3 days

## `Main.py`

The script returns the preprocessed files for all folders, then concatenates them into single dataset file, and saves the result. 
