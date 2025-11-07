import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.experimental import enable_iterative_imputer #!Do not throw it
from sklearn.impute import IterativeImputer
import os
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")


class Preprocessing():
    def __init__(self, directory_path, columns_name):
        """
        Creates new dataset class object.

        Parameters
        ----------
        directory_path : str
                        Name of the directory with all files
        columns_name : str
                       Name of the column names
        """ 
        self.directory_path = directory_path
        self.columns_name = columns_name
        
    def mini_files_to_big(self, start='2014-01-01 00:00:00', end='2020-02-29 23:00:00):
        """
        Combine all files of the directory into a single dataset.

        Parameters
        ----------
        start : str
               First time of the measurements
        end : str
               Last time of the measurements
        
        Returns
        -------
        data : pd.DataFrame
              Dataframe of combined files of the directory
        """ 
        
        # Load and concatenate dataframes of each file
        file_names = sorted(f for f in os.listdir(self.directory_path) if f.endswith('.csv'))
        df_list = [pd.read_csv(os.path.join(self.directory_path, file), header=None, sep=',') for file in file_names]
        df_concat = pd.concat(df_list, ignore_index = True)
        df_concat.columns = self.columns_name
        
        # Convert columns to numeric (float) values instead of object
        df_to_numeric = df_concat.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') #objet to float
        df_to_numeric.insert(0, 'time', pd.to_datetime(df_concat['time'])) #add the time column
        
        #Set time as index
        df_to_numeric = df_to_numeric.set_index('time')

        # Verification of duplicated raws
        # # duplicate_times = df[df.duplicated(subset=['time'], keep=False)] #!!! look after download data

        # Create full hourly date range and format the match the raw data
        df_full_range = pd.date_range(start=start, end=end, freq='h')
        
        # # Reindex initial df to fill missing timestamps with NaN
        data = df_to_numeric.reindex(df_full_range)
        data.index.name = 'time'

        return data

  
    def change_resolution(self, resolution='D')):
        """
        Resample the data to a new time resolution
        
        Parameters
        ----------
        resolution : str
                    Frequency string to resample
        
        Returns
        -------
        data : pd.DataFrame
              Dataframe with data resampled to the specified resolution
        """
        
        # Load processed data of mini_files_to_big function
        df = self.mini_files_to_big()
    
        # Calculations for columns with flux E > 0.8 and > 2 MeV 
        if 'flux' in df.columns:
            df['flux'] = df['flux']*3600 #number of particules for one hour
        if 'E08' in df.columns:
            df['E08'] = df['E08']*3600 #number of particules for one hour
        
        # Calculate He/Hp
        # if 'He' in data.columns:
        #     data['he_hp'] = data['He'] / data['Hp'] #calculate He/Hp
        #     data.drop(columns = ['He', 'Hp'], inplace = True)
        
        # Resample to another resolution
        agg_dict = {col: 'sum' for col in ['flux', 'E08'] if col in df.columns}
        agg_dict.update({col: 'mean' for col in df.columns if col not in agg_dict})
        data = df.resample(resolution).agg(agg_dict)
        
        # Calculations for flux column: replace 0 ny NaN, then convert to log10
        if 'flux' in data.columns:
            data['flux'] = np.log10(data['flux'].replace(0, np.nan))
        # if 'E08' in data.columns:
            # data['E08'] = np.log10(data['flux'].replace(0, np.nan))

        return data
  
    def fill_missing_by_interpolation(self, n=3):
        """
        Fill missing values using linear interpolation when missing segment is shorter than 3 days

        Returns
        -------
        data : pd.DataFrame
             DataFrame with filled missing values using linear interpolation
        
        """
        data = self.change_resolution().interpolate(method='linear', limit=n, limit_direction='both')
     
        return data
    
    
    def fill_missing_by_ML(self, splitting=0.8):
        """
        Fill missing values using InterativeImputer when missing segments are longer than 3 days

        Parameters
        ----------
        splitting : float
                   Ratio for splitting into training/val and testing sets
        
        Returns
        -------
        data : pd.DataFrame
              DataFrame with filled missing values using InterativeImputer
        """
        
        df = self.fill_missing_by_interpolation()
        
        # Divide into train and test samples 
        data_train = df[:int(len(df)*splitting)] #Train and val data
        data_test = df[int(len(df)*splitting):]  #Test data
        
        # Fill missing data by IterativeImputer for train and test data samples
        imputer = IterativeImputer(estimator=CatBoostRegressor(verbose=0, allow_writing_files=False, thread_count=4), max_iter=10)
        data_imputed_train = pd.DataFrame(imputer.fit_transform(data_train), columns = df.columns, index=data_train.index)
        data_imputed_test = pd.DataFrame(imputer.transform(data_test), columns = df.columns, index=data_test.index)

        # Concatenate the results
        data = pd.concat([data_imputed_train, data_imputed_test]).sort_index().reset_index()
     
        return data
   



