This folder contains the script `NN_models.py` for training neural network model and visualizing results.

The function `create_sequences_multi` creates sequences of input data and corresponding target values for time series modeling, using sliding window approach with the specified time steps.

The main function `predictions` performs the following steps:
- *data preparation*: create input features and target variables
- *data splitting*: divide dataset into training, validation and test sets
- *scaling*: standardize data
- *create sequences*: create sequences of input data using the function `create_sequences_multi`
- *model defintion*: define the model (type, layers, neurons...)
- *training*: train the model and predict the results
- *save the model*: save the model weigths
- *prediction*: predict the results
- *evaluation*: compute the prediction efficiency using R^2 score

- Model results and visualization:
  - `predictions_n_days_r`2: plot R^2 scores for each day of prediction across different models
  - `fig_predicted_VS_real`: plot the time series of predicted and actual target values
  - `corr_pred_true`: plot predicted VS measured target values for a direct comparison
  - `plot_training_history`: plot the loss curves of the training 
