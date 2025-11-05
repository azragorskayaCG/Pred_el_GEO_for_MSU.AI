This folder contains the script `ML_models.py` for training classical machine learning models and visualizing results.

The main function `predictions` performs the following steps:

- *data preparation*: create input features and target variables
- *data splitting*: divide dataset into training, validation and test sets
- *model defintion*: define the model and select the algorithm (CatBoost, XGBoost, LightGBM)
- *training and prediction*: train the model and predict the results
- *evaluation*: compute the prediction efficiency using R^2 score

The following functions generate various plot to analyze data and model results:
- Dataset and target analysis:
  - `matrix_corr`: plot the correlation matrix of the dataset
  - `event_supp_8`: plot the time series of the target and the number of events exceeding the threshold
  - `flux_train_val_test`: plot the time series of target, highlighting train, validation and test periods
  
- Model results and visualization:
  - `predictions_n_days_r`2: plot R^2 scores for each day of prediction across different models
  - `fig_feature_importance`: show the top 10 features based on importance for each prediction day
  - `shap_plot`: visualize feature importance for each day using SHAP values
  - `fig_predicted_VS_real`: plot the time series of predicted and actual target values
  - `corr_pred_true`: plot predicted VS measured target values for a direct comparison
