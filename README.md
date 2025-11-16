# General info
As part of the MSU.AI course (2025), a model predicting daily fluxes of relativistic electrons > 2 MeV at geostationary orbit with machine learning methods was developped.

The versions of the imported modules are available in the [requirements.txt](https://github.com/azragorskayaCG/requirements) file.

## Data

The initial data are downloaded from the website of the MSU Space Weather Center https://swx.sinp.msu.ru/ as time series data.

The raw data files are available in the folder [data](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data), split into 3 subfolders with distinct types of data:
- [files_geomag_indices](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_geomag_indices) contains  geomagnetic indices, listed in the following order: ``` AE-index, AU-index, AL-index, AO-index, ASY/H, SYM/D, SYM/H, ASY/D, Kp ```
- [files_solar_wind](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_solar_wind) contains the solar wind parameters, listed in the following order: ```proton density, Solar wind synamic pressure, Bulk speed, Ion tempeture, GSM B_x, GSM B_y, GSM B_z, GSM B_t ```
- [files_flux_and_others](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_flux_and_others) contains the flux and magnetosphere parameters, listed in the following order: ``` E >0.8 MeV, E >2.0 MeV, Hp, He, Hn, B, Rss(Shue) ```

Each file contains 2 months of data with an hourly resolution.

## Preprocessing

Before training, the data are preprocessed. The output consists of one file with all the parameters in daily resolution, with filled missing data.

The script is available in the folder [scripts/preprocessing](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/scripts/preprocessing).

See details in ```scripts/preprocessing/README.md ```.

## Classical machine learning

The first developed model is based on gradient boosting method of classical machine learning. Three algorithms were used:
-  ```CatBoost  ```
-  ```LightGBM  ```
-  ```XGBoost  ```

See the script in the folder [scripts/classicalML](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/scripts/classicalML), read ```scripts/classicalML/README.md ``` for more information.

## Neural networks

A second model based on LSTM neural network was developed.

See the script in the folder [scripts/NeuralNetwork](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/scripts/NeuralNetwork) and read ```scripts/NeuralNetwork/README.md ``` for more information.
