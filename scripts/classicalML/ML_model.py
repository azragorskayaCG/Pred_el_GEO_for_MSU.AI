import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, GRU, Dense, Dropout, LSTM, BatchNormalization, Bidirectional, MaxPooling1D, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
import random
import os
from matplotlib.dates import (
    DateFormatter, AutoDateLocator, AutoDateFormatter, datestr2num)
from sklearn.multioutput import MultiOutputRegressor
import shap
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.simplefilter("ignore", FutureWarning)


# ================= 
# ================= 

# MODEL

def set_seed(seed=42):
    random.seed(seed)                     
    np.random.seed(seed)            
    tf.random.set_seed(seed)             
    os.environ['PYTHONHASHSEED'] = str(seed) 

set_seed(42)

def predictions(data):
    """
    Script for training classical machine learning models, structured into the following steps: 
    - Data preparation 
    - Data splitting (train, validation, test)
    - Model defintion
    - Training and prediction
    - Evaluation
    
    Parameters
    -----------
    data: pd.DataFrame containing all parameters with daily resolution.
    """
    
    #Data preparation
    data = data.copy()
    data['time'] = pd.to_datetime(data['time'].str.slice(0, 10))
    data.set_index('time', inplace=True)
    target = 'flux'
    
    cols_to_use_all = ['Vsw', 'P', 'kp', 'E08', 'iontemp', 'He', 'bz', 'bt']
    others = sorted(cols_to_use_all)
    
    cols_to_use = [target] + others 
    
    # Create lag values
    lags = np.arange(1, 3)

    for col in cols_to_use:
        for lag in lags:
            data[f'{col}_day_-{lag}'] = data[col].shift(lag)
    
    lag_cols = [f'{col_name}_day_-{lag}' for col_name in cols_to_use for lag in lags]
    
    # Create 3-step targets
    data['target_day_1'] = data[target].shift(0)
    data['target_day_2'] = data[target].shift(-1)
    data['target_day_3'] = data[target].shift(-2)
    data.dropna(inplace=True)
    
    x = data[lag_cols]
    y = data[['target_day_1', 'target_day_2', 'target_day_3']]

    time_steps = 2
    
    # Data splitting
    x_train, x_val_test, y_train, y_valtest = train_test_split(x, y, test_size = 0.3, shuffle = False)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_valtest, test_size = 0.5, shuffle = False)
    
    # Model defintion
    
    # CatBoost model
    model_base = CatBoostRegressor(
        verbose=0,
        allow_writing_files=False,
        n_estimators=500,
        l2_leaf_reg=10,
        depth = 7,
        colsample_bylevel=0.9,
    )
    
    # XGBoost model
    # model_base = XGBRegressor(
    #     n_estimators=100,
    #     max_depth=7,
    #     reg_lambda=10,            # L2 регуляризация
    #     # colsample_bylevel=0.9,   # Аналог colsample_bylevel в CatBoost
    #     verbosity=0,
    #     use_label_encoder=False   # Чтобы избежать предупреждений
    # )
    
    # LightGBM model
    # model_base = LGBMRegressor(
    #     n_estimators=100,
    #     max_depth=5,
    #     reg_alpha=10,     
    #     colsample_bytree=0.9,    # Аналог colsample_bylevel применяется к выбору признаков на уровне дерева
    #     verbose=-1
    # )
    
    model = MultiOutputRegressor(model_base)
    
    # Training and prediction
    #Train
    model.fit(x_train, y_train)
    
    # Save the history
    
    histories = []

    # Predict values
    y_pred = model.predict(x_test)

    # Evaluation
    pe_all = []
    for h in range(3):
        pe_all.append(r2_score(y_test.iloc[:, h], pd.DataFrame(y_pred).iloc[:, h]))
    
    test_index = data.index[time_steps - 1 + len(x_train) + len(x_val):]

    
    return pd.DataFrame(y_pred), pe_all, y_test, test_index, model, histories, x_test

# ==============
# FIGURES
# ==============

# Dataset and target analysis

def matrix_corr(data):
    """
    Plot the correlation matrix of the dataset
    """
    
    # Calculate the correlation matrix
    cols_to_use_all = ['Vsw', 'P', 'kp', 'E08', 'rss', 'iontemp', 'He', 'bz', 'bt']
    others = sorted(cols_to_use_all)
    cols_to_use = ['flux'] + others
    data_to_matrix = data[cols_to_use]
    
    method = 'spearman'
    correlation_matrix = data_to_matrix.corr(method=method) #spearman, pearson
    
    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Spearman Correlation Matrix')
    plt.grid(False)
    # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/corr_matrix_{method}.png', dpi = 600, bbox_inches = 'tight')
    
    plt.show()

def event_supp_8(data):
    """
    Plot the time series of the target and the number of events exceeding the threshold
    """
    
    fig, axs = plt.subplots(2, 1, sharex = True, gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(10) 
    fig.set_figheight(8) 
    
    (ax1), (ax2) = axs
    
    data_for_plot = data.copy()
    data_for_plot['year'] = data_for_plot['time'].str.slice(0,4)
    data_for_plot = data_for_plot[data_for_plot['flux']>8]
    year_sup_8 = data_for_plot.groupby('year').size()
    
    ax1.plot(data.index, data['flux'], color = 'royalblue')
    ax1.axhline(y=8, xmin=0, xmax=1, color = 'darkred', linestyle = '--', linewidth = 3)
    ax2.plot(data_for_plot.groupby('year').apply(lambda x: x.index[0]), year_sup_8.values, color = 'darkred')
    ax2.scatter(data_for_plot.groupby('year').apply(lambda x: x.index[0]), year_sup_8.values, s = 100, color = 'darkred')
    
    ax1.set_ylabel('$log_{10}(J_{el > 2 МэВ})$')
    ax2.set_ylabel('$N_{событий}$')
    ax2.set_xlabel('Год')
    
    ax2.set_xticks(data_for_plot.groupby('year').apply(lambda x: x.index[0]), labels = year_sup_8.index)
    
    ax1.text(2250, 9.5, 'a', fontstyle = 'italic')
    ax2.text(2250, 180, 'б', fontstyle = 'italic')

    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/event_supp_8.png', dpi = 600, bbox_inches = 'tight')

    plt.show()
    
def flux_train_val_test(data):
    """
    Plot the time series of target, highlighting train, validation and test periods
    """
    fig, axs = plt.subplots(1, 1, sharex = True, gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(8) 
    fig.set_figheight(5) 
    
    fontsize_here = 25
    data.set_index(pd.to_datetime(data['time']), inplace = True)
    train_data, other_data = train_test_split(data['flux'], test_size = 0.3, shuffle = False)
    val_data, test_data = train_test_split(other_data, test_size = 0.5, shuffle = False)

    # ax1.plot(pd.to_datetime(data['time']), data['flux'], color = 'royalblue')
    plt.plot(train_data.index, train_data, label = 'Обучение\n(максимум + спад)')
    plt.plot(val_data.index, val_data, label = 'Валидация\n(спад + минимум)')
    plt.plot(test_data.index, test_data, label = 'Тестирование\n(минимум)')

    plt.legend(bbox_to_anchor=(1.0, 0.84))
    plt.ylabel('$log_{10}(J_{el > 2 МэВ})$', fontsize = fontsize_here)
    ticks = pd.date_range(start=data.index[0], end=data.index[-1], freq='YS') 
    plt.xticks(ticks, [t.year for t in ticks], rotation=35, fontsize=fontsize_here)
    plt.yticks(fontsize = fontsize_here)
    plt.xlabel('Год', fontsize = fontsize_here)
    plt.xlim(data.index[0], data.index[-1])
    

    plt.ylim(4.5, 10)
    
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/flux_train_val_test.png', dpi = 600, bbox_inches = 'tight')

    plt.show()
    
# =============
# MODEL RESULTS AND VISUALIZATION

def predictions_n_days_r2():
    """
    Plot R^2 scores for each day of prediction across different models
    """
    
    plt.scatter(np.arange(1, len(pe)+1), [0.89, 0.71, 0.53], label = f'LSTM        0.89, 0.71, 0.53', color = 'green')
    plt.plot(np.arange(1, len(pe)+1), [0.89, 0.71, 0.53], color = 'green', linestyle = '--')
    plt.scatter(np.arange(1, len(pe)+1), [0.86, 0.69, 0.52], label = f'CatBoost  0.86, 0.69, 0.52', color = 'dodgerblue')
    plt.plot(np.arange(1, len(pe)+1), [0.86, 0.69, 0.52], color = 'dodgerblue')
    plt.scatter(np.arange(1, len(pe)+1), [0.85, 0.68, 0.50], label = f'LightGBM 0.85, 0.68, 0.50', color = 'orange')
    plt.plot(np.arange(1, len(pe)+1), [0.85, 0.68, 0.50], color = 'orange')
    plt.scatter(np.arange(1, len(pe)+1), [0.83, 0.64, 0.46], label = f'XGBoost   0.83, 0.64, 0.46', color = 'darkred')
    plt.plot(np.arange(1, len(pe)+1), [0.83, 0.64, 0.46], color = 'darkred')

    # plt.scatter(np.arange(1, len(pe)+1), pe, label = f'{pe[0]:.2f}, {pe[1]:.2f}, {pe[2]:.2f}')
    # plt.plot(np.arange(1, len(pe)+1), pe)
    
    # plt.yticks(np.arange(0.45, 0.95, 0.1))
    plt.xticks(np.arange(1, 4))
    plt.xlabel('День прогнозирования')
    plt.ylabel('R²')
    # plt.title('')
    plt.legend(loc = 'upper right', fontsize = 11)
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/r2.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    
def fig_feature_importance(model):
     """
     Show the top 10 features based on importance for each prediction day
     """
    
    all_names, all_percent = [], []
    for i, estimator in enumerate(model.estimators_):
        feature_importances = estimator.get_feature_importance()
        feature_names = estimator.feature_names_
    
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_idx][:10]
        sorted_names = np.array(feature_names)[sorted_idx][:10]
        importances_percent = 100 * sorted_importances / sorted_importances.sum()
        all_names.append(sorted_names)
        all_percent.append(importances_percent)
    
    fig, axs = plt.subplots(3, 1, sharex = True, gridspec_kw={'hspace': 0.05, 'wspace': 0.05}) #row, col
    fig.set_figwidth(6) 
    fig.set_figheight(14) 

    (ax1), (ax2), (ax3) = axs
    
    ax1.barh(all_names[0], all_percent[0], zorder=3, color = 'green', label = 'Day 1')
    ax2.barh(all_names[1], all_percent[1], zorder=3, color = 'orange', label = 'Day 2')
    ax3.barh(all_names[2], all_percent[2], zorder=3, color = 'red', label = 'Day 3')
    
    ax3.set_xlabel('%')
    
    ax1.set_title('Top 10 feature importance')
    ax1.set_title('Top 10 feature importance')
    ax1.set_title('Top 10 feature importance')

    ax1.legend(loc = 'lower right')
    ax2.legend(loc = 'lower right')
    ax3.legend(loc = 'lower right')

    ax1.set_xticks(np.arange(0, 60, 10))
    
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/feature_importance.png', dpi = 600, bbox_inches = 'tight')

    plt.show()

def shap_plot(model):
    """
    Visualize feature importance for each day using SHAP values
    """
    
    explanations = []
    for i, estimator in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(x_test)
        explanations.append(shap_values)
        
    for i, shap_values in enumerate(explanations):
        # Step 1: Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        
        # Step 2: Normalize to sum to 1 (proportions)
        proportions = mean_abs_shap / mean_abs_shap.sum()
        
        # Step 3: Create a new Explanation object with normalized values
        explanation_norm = shap.Explanation(
            values=proportions*100,
            base_values=np.zeros_like(proportions),
            data=None,
            feature_names=shap_values.feature_names
        )
        
        # Step 4: Plot normalized SHAP values as a bar plot
        # shap.summary_plot(shap_values.values, x_test, feature_names=x_test.columns, show=False) #multicolor beautiful bars
        # shap.summary_plot(explanations[i].values, x_test, plot_type="bar", show=False) #all bars of feature importances
        # shap.summary_plot(shap_values, plot_type='violin', show=False) #violon
        # shap.plots.bar(shap_values, show=False) #bars with + sum of N features
   
        shap.plots.bar(explanation_norm, show=False)
        
        plt.title(f"День {i + 1}")
        
        plt.xlim(0, 35)
        
        plt.xlabel('Важность, %', fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        
        # Uncomment to save the figures if needed
        # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/shap_norm_{i+1}.png', dpi=600, bbox_inches='tight')
        
        # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/shap/colored_beautiful_bars_{i+1}.png', dpi = 600, bbox_inches = 'tight')
        # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/shap/all_bar_{i+1}.png', dpi = 600, bbox_inches = 'tight')
        # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/shap_{i+1}.png', dpi = 600, bbox_inches = 'tight')
        # plt.savefig(f'/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/shap/violon_{i+1}.png', dpi = 600, bbox_inches = 'tight')

        plt.show()
    
def fig_predicted_VS_real(future_predictions, y_test, pe):
    """
    Plot the time series of predicted and actual target values
    """
    fig, axs = plt.subplots(3, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(10) 
    fig.set_figheight(8) 

    (ax1), (ax2), (ax3) = axs

    ax1.plot(y_test.index, future_predictions[0], color = 'green', linewidth = 1)
    ax1.plot(y_test.index, y_test['target_day_1'], color = 'black', linewidth = 1, label = 'True')

    ax2.plot(y_test.index, future_predictions[1], color = 'orange', linewidth = 1)
    ax2.plot(y_test.index, y_test['target_day_2'], color = 'black', linewidth = 1)

    ax3.plot(y_test.index, future_predictions[2], color = 'red', linewidth = 1)
    ax3.plot(y_test.index, y_test['target_day_3'], color = 'black', linewidth = 1)

    label = ax1.set_ylabel('$log_{10}$ Electron flux')
    ax1.yaxis.set_label_coords(-0.1, -0.5)
    ax1.set_xlim(y_test.index[0], y_test.index[-1])
    ax1.set_title('Actual and predicted values')
    ax1.legend(loc = 'upper left')
    
    ax1.text(datestr2num('2020-01-20'), 9., 'Day 1', color = 'green')
    ax2.text(datestr2num('2020-01-20'), 9., 'Day 2', color = 'orange')
    ax3.text(datestr2num('2020-01-20'), 9., 'Day 3', color = 'red')

    fig.autofmt_xdate(rotation=45)
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/true_VS_pred_in_time.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
def corr_pred_true(future_predictions, y_test, pe):
    """
    Plot predicted VS measured target values for a direct comparison
    """ 
    fig, axs = plt.subplots(3, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(4) 
    fig.set_figheight(10) 

    (ax1), (ax2), (ax3) = axs

    ax1.scatter(future_predictions[0], y_test['target_day_1'], color = 'green', s = 10, label = 'Day 1')
    ax1.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    ax2.scatter(future_predictions[1], y_test['target_day_2'], color = 'orange', s = 10, label = 'Day 2')
    ax2.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    ax3.scatter(future_predictions[2], y_test['target_day_3'], color = 'red', s = 10, label = 'Day 3')
    ax3.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    label = ax1.set_ylabel('Measured $log_{10}$ electron flux')
    ax1.yaxis.set_label_coords(-0.15, -0.5)

    ax1.set_title('Actual VS predicted values', fontsize = 19)
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(loc='lower right')

    ax1.set_ylim(4, 10.5)
    ax2.set_ylim(4, 10.5)
    ax3.set_ylim(4, 10.5)
    
    ax1.set_xlim(4., 10.5)
    ax1.set_xticks(ticks = np.arange(4, 11, 2))

    ax3.set_xlabel('Predicted $log_{10}$ electron flux')
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/ML_model/true_VS_pred_in_corr.png', dpi = 600, bbox_inches = 'tight')
    plt.show()

# def plot_training_history(history):
#     # Loss
#     plt.figure(figsize=(7, 5))
#     plt.plot(history['target_day_1']['learn']['RMSE'], label='Train')
#     plt.plot(history['target_day_1']['validation']['RMSE'], label='Val')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMSE')
#     plt.title('Training/Validation RMSE', fontsize = 19)
#     plt.legend()

#     # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/NeuralNetwork/loss_curves.png', dpi = 600, bbox_inches = 'tight')
#     plt.show()
    

    
# =================
OPEN THE DATASET AND PROCESSING
# ================= 

dirname = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Processed/'
data = pd.read_csv(dirname + 'dataset_final.csv')

# =========== 
# Processing

predicted_results = predictions(data)
future_predictions, pe, y_test, index, model, history, x_test = predicted_results[0], predicted_results[1], predicted_results[2],  predicted_results[3], predicted_results[4], predicted_results[5], predicted_results[6]

# =========== 
# PLOT THE RESULT
# =========== 

plt.rcParams.update({'font.size': 20, 'axes.grid': True, 'xtick.direction': 'in', 'ytick.direction': 'in'})

# =========== 

# matrix_corr(data)
# event_supp_8(data)
# flux_train_val_test(data)

# predictions_n_days_r2() #R^2
# fig_feature_importance(model) #Importance of each parameter

# fig_predicted_VS_real(future_predictions, y_test, pe) #Trace the profile of values in time
# corr_pred_true(future_predictions, y_test, pe) #Measured VS Predicted

shap_plot(model)

