from Preprocessing import *

directory_path = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Preprocessing/All_small_files/files_flux_and_others/'
directory_path_2 = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Preprocessing/All_small_files/files_geomag_indices/'
directory_path_3 = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Preprocessing/All_small_files/files_solar_wind/'

## === Processed data

# = Main features
results_flux_etal = Preprocessing(directory_path, ['time', 'E08', 'flux', 'Hp', 'He', 'Hn', 'B', 'rss'])
final_flux_etal = results_flux_etal.predict_more_than_n()

# = Added features
results_geomag_indices = Preprocessing(directory_path_2, ['time', 'AE', 'AU', 'AL', 'AO', 'asyh', 'symd', 'symh', 'asyd', 'kp'])
final_geomag_indices = results_geomag_indices.predict_more_than_n()

results_solar_wind = Preprocessing(directory_path_3, ['time', 'pdensity', 'P', 'Vsw', 'iontemp', 'bx', 'by', 'bz', 'bt'])
final_solar_wind = results_solar_wind.predict_more_than_n()

# # # = Concatenate main data and added

data1 = pd.merge(final_flux_etal, final_geomag_indices, on='time', how='left')
data = pd.merge(data1, final_solar_wind, on='time', how='left')


# == Save data

# data.to_csv('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Processed/dataset_final.csv', index = False)
