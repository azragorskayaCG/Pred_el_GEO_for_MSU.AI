# General info
As part of the MSU.AI course (2025), a model predicting daily fluxes of relativistic electrons > 2 MeV at geostationary orbit with machine learning methods was developped.

## Data

The initial data are downloaded from the site of the MSU Space Weather Center https://swx.sinp.msu.ru/ with an hourly resolution.
The raw data files are available in the folder [data](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data).

The folder data contains 3 subfolders with distinct types of data:
- [files_geomag_indices](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_geomag_indices) contains  geomagnetic indices, including: ``` AE-index, AU-index, AL-index, AO-index, ASY/H, SYM/D, SYM/H, ASY/D, Kp ```
- [files_solar_wind](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_solar_wind) contains the solar wind parameters, including: ```proton density, Solar wind synamic pressure, Bulk speed, Ion tempeture, GSM B_x, GSM B_y, GSM B_z, GSM B_t ```
- [files_flux_and_others](https://github.com/azragorskayaCG/Pred_el_GEO_for_MSU.AI/tree/main/data/files_flux_and_others) contains the flux and magnetosphere parameters, including: ``` E >0.8 MeV, E >2.0 MeV, Hp, He, Hn, B, Rss(Shue) ```

## Processing

