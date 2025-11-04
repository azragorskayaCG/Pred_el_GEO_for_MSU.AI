import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.experimental import enable_iterative_imputer #!Do not throw it
from sklearn.impute import IterativeImputer
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")


class Preprocessing():
    
    def __init__(self, directory_path, columns_name):
        self.directory_path = directory_path
        self.columns_name = columns_name
        self._data_day = None
        self._time = None
        self._data_imputed = None
        
    def mini_files_to_big(self):
    
        file_names = sorted(f for f in os.listdir(self.directory_path) if f.endswith('.csv'))
        dfs = [pd.read_csv(os.path.join(self.directory_path, file), header=None, sep=',') for file in file_names]
        df = pd.concat(dfs, ignore_index = True)
        df.columns = self.columns_name
        df_numeric = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce') #objet to float
        df_numeric.insert(0, 'time', df['time'].values)
        
        # duplicate_times = df[df.duplicated(subset=['time'], keep=False)] #!!! look after download data

        # Filled missing data time by NaN
        full_time = pd.date_range(start='2014-01-01 00:00:00', end = '2020-02-29 23:00:00', freq = 'h')
        full_time = full_time.astype(str) + ".000   "
        df_numeric = df_numeric.set_index('time')
        df_full = df_numeric.reindex(full_time)
        df_full.index.name = 'time'
        df_full = df_full.reset_index()

        return df_full

  
    def pre_calculations(self):
        
        # == Open file
        
        df_numeric = self.mini_files_to_big()
        data = df_numeric.iloc[:, 1:].copy()
        time =  df_numeric['time'][23::24]
        
        # == Pre calculations        
        
        if 'flux' in data.columns:
            data['flux'] = data['flux']*3600 #number of particules for one hour
        if 'E08' in data.columns:
            data['E08'] = data['E08']*3600 #number of particules for one hour

        # if 'He' in data.columns:
        #     data['he_hp'] = data['He'] / data['Hp'] #calculate He/Hp
        #     data.drop(columns = ['He', 'Hp'], inplace = True)
        
        # == Calculate the daily values
     
        agg_dict = {col: 'sum' for col in ['flux', 'E08'] if col in data.columns}
        agg_dict.update({col: 'mean' for col in data.columns if col not in agg_dict})
           
        data_day = data.groupby(data.index // 24).agg(agg_dict)
        
        if 'flux' in data.columns:
            data_day['flux'] = data_day['flux'].replace(0, np.nan)
            data_day['flux'] = np.log10(data_day['flux']) #convert flux to log10 
            data_day.insert(len(data_day.columns)-1, 'flux', data_day.pop('flux'))
            
        # if 'E08' in data.columns:
        #     data_day['E08'] = data_day['E08'].replace(0, np.nan)
        #     data_day['E08'] = np.log10(data_day['E08']) #convert flux to log10 
        #     data_day.insert(len(data_day.columns)-1, 'E08', data_day.pop('E08'))
        
        self._data_day = data_day
        self._time = time
     
        return data_day, time
  
    def interpolate_fewer_than_n(self, n=3):
        df_interp = self.pre_calculations()[0].copy()
        df = df_interp.interpolate(method='linear', limit=n, limit_direction='both')
     
        return df
    
    
    def predict_more_than_n(self):
     
        data = self.interpolate_fewer_than_n()
        
        data_train = data.copy()[:int(len(data)*0.8)] #Divide into train data
        data_test = data.copy()[int(len(data)*0.8):]  #Divide into test data
        
        imputer = IterativeImputer(estimator=CatBoostRegressor(verbose=0, allow_writing_files=False, thread_count=4), max_iter=10)
        data_imputed_train = pd.DataFrame(imputer.fit_transform(data_train), columns = data.columns, index=data_train.index)
        data_imputed_test = pd.DataFrame(imputer.transform(data_test), columns = data.columns, index=data_test.index)

        data_imputed = pd.concat([data_imputed_train, data_imputed_test]).sort_index()
        data_imputed['time'] = self.pre_calculations()[1].values
     
        
        return data_imputed



