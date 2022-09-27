# Preprocessor Specification
## Introduction
This directory contains a series of python scripts that perform data preprocessing tasks as well a a driver program that will organize a preprocessing pipeline.

#### Functions
1. Importing csv data
2. Removing rows and columns
3. Flattening data using juction tables
4. Imputing missing values
5. Transforming/engineering new features
6. Export in-memory data back to csv

## Scripts
#### csv.py
Facilitates import and export operations between csv data files and pandas dataframes.

#### filter.py
Handles the removal of rows and columns.

#### transform.py
Contains functions that transforms features, combines existing features into new ones, and impute missing values.

#### preprocess.py
Driver program that laces together a data preprocessing pipeline using the other helper scripts.