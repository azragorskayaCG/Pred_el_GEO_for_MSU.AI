This folder contains scripts for data preprocessing.

## `Preprocessing.py`

The `Preprocessing` class has functions for preprocessing all files within a directory:

- `mini_files_to_big`: combine all files of the directory into a single dataset
- `pre_calculations`: convert hourly data to daily resolution
- `interpolate_fewer_than_n`: fill missing values using linear interpolation when missing segment is shorter than 3 days 
- `predict_more_than_n`: fill missing values using InterativeImputer when missing segments are longer than 3 days

## `Main.py`

Return the preprocessed files for all folders, and concatenates them into single dataset file, and saves the result. 
