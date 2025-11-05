import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import numpy as np
import tensorflow as tf
import shap
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
# ================= 
# ================= 


def set_seed(seed=42):
    random.seed(seed)                     
    np.random.seed(seed)            
    tf.random.set_seed(seed)             
    os.environ['PYTHONHASHSEED'] = str(seed) 

set_seed(42)

def create_sequences_multi(x, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(x) - time_steps):
        X_seq.append(x.iloc[i:i+time_steps].values)
        y_seq.append(y.iloc[i+time_steps])
    return np.array(X_seq), np.array(y_seq)


def predictions(data):
    data = data.copy()
    data['time'] = pd.to_datetime(data['time'].str.slice(0, 10))
    data.set_index('time', inplace=True)
    target = 'flux'
    
    cols_to_use_all = ['Vsw', 'P', 'kp', 'E08', 'rss', 'iontemp', 'He', 'bz', 'bt']
    others = sorted(cols_to_use_all)
    cols_to_use = [target] + others 
    
    # Create 3-step targets
    data['target_day_1'] = data[target].shift(0)
    data['target_day_2'] = data[target].shift(-1)
    data['target_day_3'] = data[target].shift(-2)
    data.dropna(inplace=True)
    
    x = data[cols_to_use]
    y = data[['target_day_1', 'target_day_2', 'target_day_3']]
    
    time_steps = 2
    
    # ======= Explicit train/val/test split =======
    # 80% train+val, 20% test first
    x_train_raw, x_valtest_raw, y_train_raw, y_valtest_raw = train_test_split(
        x, y, test_size=0.3, shuffle=False)
        
    x_val_raw, x_test_raw, y_val_raw, y_test_raw = train_test_split(
        x_valtest_raw, y_valtest_raw, test_size=0.5, shuffle=False)
    
    # ======= Scaling =======
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    x_train_scaled = pd.DataFrame(scaler_x.fit_transform(x_train_raw), 
                                  columns=x_train_raw.columns, index=x_train_raw.index)

    x_val_scaled = pd.DataFrame(scaler_x.transform(x_val_raw), 
                                columns=x_train_raw.columns, index=x_val_raw.index)
    x_test_scaled = pd.DataFrame(scaler_x.transform(x_test_raw), 
                                 columns=x_train_raw.columns, index=x_test_raw.index)
    
    y_train_scaled = pd.DataFrame(scaler_y.fit_transform(y_train_raw), 
                                  columns=y.columns, index=y_train_raw.index)
    y_val_scaled = pd.DataFrame(scaler_y.transform(y_val_raw), 
                                columns=y.columns, index=y_val_raw.index)
    y_test_scaled = pd.DataFrame(scaler_y.transform(y_test_raw), 
                                 columns=y.columns, index=y_test_raw.index)
    
    # ======= Create sequences =======
    x_train_seq, y_train_seq = create_sequences_multi(x_train_scaled, y_train_scaled, time_steps)
    x_val_seq, y_val_seq = create_sequences_multi(x_val_scaled, y_val_scaled, time_steps)
    x_test_seq, y_test_seq = create_sequences_multi(x_test_scaled, y_test_scaled, time_steps)
    
    # ======= Prepare y sequences as list for multi-output =======
    y_train_seq = [y_train_seq[:, i] for i in range(3)]
    y_val_seq = [y_val_seq[:, i] for i in range(3)]
    
    # ======= Build model (same as before) =======
    inputs = Input(shape=(time_steps, x_train_raw.shape[1]))
    
    x = LSTM(20, activation='tanh', return_sequences=True)(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(20, activation='tanh', return_sequences=True)(x)
    x = LSTM(20, activation='tanh')(x)

    out1 = Dense(1, name='day_1')(x)
    out2 = Dense(1, name='day_2')(x)
    out3 = Dense(1, name='day_3')(x)
    
    model = Model(inputs=inputs, outputs=[out1, out2, out3])
    model.compile(optimizer=Adam(),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]*3)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.0001
    )
    
    # ======= Fit model with explicit validation data =======
    history = model.fit(
        x_train_seq, y_train_seq,
        validation_data=(x_val_seq, y_val_seq),
        epochs=200,
        batch_size=64,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # model.save('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/model_NN/my_model.keras')
        
    # ======= Predict on test =======
    y_pred_scaled = model.predict(x_test_seq, verbose=0)
    y_pred_scaled_stacked = np.hstack(y_pred_scaled)  # shape (samples, 3)
    
    # Align y_test_scaled to match sequence length
    y_test_scaled_aligned = y_test_scaled.iloc[time_steps:].values
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled_stacked)
    y_true = scaler_y.inverse_transform(y_test_scaled_aligned)
    
    pe_all = []
    for h in range(3):
        pe_all.append(r2_score(y_test_raw.iloc[time_steps:, h], y_pred[:, h]))
    
    test_index = data.index[time_steps - 1 + len(x_train_raw) + len(x_val_raw):]
    
    # Values of transform
    # x_transform_params = pd.DataFrame({
    # 'x_min': scaler_x.data_min_,
    # 'x_range': scaler_x.data_range_,
    # }, index = x_train_raw.columns)
    
    # x_transform_params.to_csv('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/coeff_standard/coeff_values.csv')

    return y_pred, pe_all, y_test_raw, test_index, model, history

# ==============
# ==============

# PLOT THE RESULTS

def predictions_n_days_r2():
    plt.scatter(np.arange(1, len(pe)+1), pe, label = f'{pe[0]:.2f}, {pe[1]:.2f}, {pe[2]:.2f}')
    plt.plot(np.arange(1, len(pe)+1), pe)
    plt.yticks(np.arange(0.5, 0.95, 0.1))
    plt.xticks(np.arange(1, 4))
    plt.xlabel('Days ahead')
    plt.ylabel('Prediction efficiency (R²)')
    plt.title('NN 3 LSTM model')
    plt.legend(loc = 'lower left')
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Models/NN_model/PE_values.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    

def fig_predicted_VS_real(future_predictions, y_test, pe):
    fig, axs = plt.subplots(3, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(10) 
    fig.set_figheight(8) 

    (ax1), (ax2), (ax3) = axs

    ax1.plot(y_test.iloc[2:].index, future_predictions[:,0], color = 'green', linewidth = 1)
    ax1.plot(y_test.iloc[2:].index, y_test['target_day_1'].iloc[2:], color = 'black', linewidth = 1, label = 'True')

    ax2.plot(y_test.iloc[2:].index, future_predictions[:,1], color = 'orange', linewidth = 1)
    ax2.plot(y_test.iloc[2:].index, y_test['target_day_2'].iloc[2:], color = 'black', linewidth = 1)

    ax3.plot(y_test.iloc[2:].index, future_predictions[:,2], color = 'red', linewidth = 1)
    ax3.plot(y_test.iloc[2:].index, y_test['target_day_3'].iloc[2:], color = 'black', linewidth = 1)

    label = ax1.set_ylabel('$log_{10}$(J)')
    ax1.yaxis.set_label_coords(-0.1, -0.5)
    ax1.set_xlim(y_test.index[0], y_test.index[-1])
    # ax1.set_title('Actual and predicted values')
    ax1.legend(loc = 'upper left')
    
    ax1.text(datestr2num('2019-12-24'), 9., 'День 1', color = 'green')
    ax2.text(datestr2num('2019-12-24'), 9., 'День 2', color = 'orange')
    ax3.text(datestr2num('2019-12-24'), 9., 'День 3', color = 'red')

    fig.autofmt_xdate(rotation=45)
    
    ax1.text(datestr2num('2020-02-10'), 9., 'а', fontstyle = 'italic', fontsize = 23)
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/true_VS_pred_in_time.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
def corr_pred_true(future_predictions, y_test, pe):
    fig, axs = plt.subplots(3, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0.05}) #row, col
    fig.set_figwidth(5) 
    fig.set_figheight(10) 

    (ax1), (ax2), (ax3) = axs

    ax1.scatter(y_test['target_day_1'].iloc[2:], future_predictions[:, 0], color = 'green', s = 10, label = 'День 1')
    ax1.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    ax2.scatter(y_test['target_day_2'].iloc[2:], future_predictions[:, 1], color = 'orange', s = 10, label = 'День 2')
    ax2.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    ax3.scatter( y_test['target_day_3'].iloc[2:], future_predictions[:, 2], color = 'red', s = 10, label = 'День 3')
    # ax3.scatter(future_predictions[2], pd.Series(y_test).shift(2).dropna(), color = 'red', linewidth = 1, label = 'Day 3')
    ax3.plot(np.arange(4, 12, 1), np.arange(4, 12, 1), color = 'black', linewidth = 2)

    label = ax1.set_ylabel('Прогнозируемые $log_{10}$(J)')
    ax1.yaxis.set_label_coords(-0.15, -0.5)

    # ax1.set_title('Actual VS predicted values', fontsize = 19)
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(loc='lower right')

    ax1.set_ylim(4, 10.5)
    ax2.set_ylim(4, 10.5)
    ax3.set_ylim(4, 10.5)
    
    ax1.set_xlim(4., 10.5)
    ax1.set_xticks(ticks = np.arange(4, 11, 2))

    ax3.set_xlabel('Измеренные $log_{10}$(J)')
    
    ax1.text(4.4, 9.5, 'б', fontstyle = 'italic', fontsize = 23)
    # fig.subplots_adjust(bottom=0.3)

    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/Article_presentation/true_VS_pred_in_corr.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    
def plot_training_history(history):
    # Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss', fontsize = 19)
    plt.legend()

    # Metrics (Root Mean Squared Error for each output)
    plt.subplot(1, 2, 2)
    metric_name = 'day_3_root_mean_squared_error'
    val_metric_name = 'val_day_3_root_mean_squared_error'
    if metric_name in history.history:
        plt.plot(history.history[metric_name], label='Train')
    if val_metric_name in history.history:
        plt.plot(history.history[val_metric_name], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training/Validation RMSE', fontsize = 19)
    plt.legend()
    
    plt.subplots_adjust(wspace=0.3)
    # plt.savefig('/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/Figures/NeuralNetwork/loss_curves.png', dpi = 600, bbox_inches = 'tight')
    plt.show()
    
# ================= 
# ================= 

dirname = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/NeuralNetwork/data/Processed/'

# =========== 

data = pd.read_csv(dirname + 'dataset_final.csv')

# =========== 

predicted_results = predictions(data)
future_predictions, pe, y_test, index, model, history = predicted_results[0], predicted_results[1], predicted_results[2],  predicted_results[3], predicted_results[4], predicted_results[5]

# =========== 
# =========== 

plt.rcParams.update({'font.size': 20, 'axes.grid': True, 'xtick.direction': 'in', 'ytick.direction': 'in'})

# =========== 

predictions_n_days_r2() #R^2

# fig_predicted_VS_real(future_predictions, y_test, pe) #Trace the profile of values in time
# corr_pred_true(future_predictions, y_test, pe) #Measured VS Predicted

# plot_training_history(history) #Loss and RMSE curves for train and val

aa = y_test['target_day_1'].iloc[2:]-future_predictions[:,0]
plt.scatter(y_test['target_day_1'].iloc[2:], aa)
plt.axhline(y = 0, xmin = 0, xmax = 1, color = 'black')
plt.show()

plt.hist(aa)
plt.show()


