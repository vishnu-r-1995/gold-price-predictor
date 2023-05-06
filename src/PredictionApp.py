import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from keras.optimizers.adam import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sb

def get_dataframe_for_ticker(ticker, time_period):
    dataframe = yf.Ticker(ticker).history(period = time_period).reset_index()
    return dataframe

def get_csv_dataframe(file_path):
    return pd.read_csv(file_path)

def get_correlation_matrix_for_dataframe(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

def write_dataframe_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path)

def write_to_file(str, path):
    f = open(path, "a")
    f.write(str)
    f.close()

def get_combined_dataframes_using_date(dataframe_a, dataframe_b):
    dataframe_a['Inflation'] = 0
    for i in range(len(dataframe_a)):
        datetime_str_a = dataframe_a.iloc[i].Date
        date_str_a = datetime_str_a.split(' ')[0]
        datetime_obj_a = datetime.strptime(date_str_a, '%Y-%m-%d')
        for j in range(len(dataframe_b)):
            date_str_b = dataframe_b.iloc[j].DATE
            datetime_obj_b = datetime.strptime(date_str_b, '%Y-%m-%d')
            if (datetime_obj_a.date() == datetime_obj_b.date()):
                dataframe_a.at[i, 'Inflation'] = dataframe_b.iloc[j]['T10YIE']
                break
    return dataframe_a   
        
def get_dataframe_with_replaced_value(dataframe, column_name, from_value, to_value):
    dataframe[column_name].mask(dataframe[column_name] == from_value, to_value, inplace=True)
    return dataframe
  
tf.random.set_seed(42)
ticker = 'GC=F'
time_period = '20y'
dataframe_for_gold_close_prediction = get_csv_dataframe('C:\All\Dataset\gold_inflation_crude_dji_wmt_int_cs.csv');
dataframe_for_gold_close_prediction = dataframe_for_gold_close_prediction.iloc[: -825, :]
dataframe_for_gold_close_prediction = dataframe_for_gold_close_prediction.reset_index()
dataframe_columns_used = ['Open', 'High', 'Low', 'DJI Adj Close', 'Interest Rate', 'Consumer Sentiment',
                         'WMT Adj Close', 'Crude Oil Adj Close', 'Inflation', 'Close']
number_of_features = len(dataframe_columns_used) - 1
data_file_path = "C:\All\Dataset\Files\Data.txt"

def display_scatter_matrix_for_dataframe(dataframe, list_of_columns):
    scatter_matrix(dataframe[list_of_columns], figsize = (12, 8))
    plt.show()
    
correlation_matrix = get_correlation_matrix_for_dataframe(dataframe_for_gold_close_prediction[dataframe_columns_used])

display_scatter_matrix_for_dataframe(dataframe_for_gold_close_prediction, 
                                     ['Close', 'Crude Oil Adj Close', 'Inflation', 'DJI Adj Close', 
                                      'WMT Adj Close', 'Interest Rate', 'Consumer Sentiment'])
print(correlation_matrix['Close'].sort_values(ascending = False))
write_dataframe_to_csv(correlation_matrix['Close'].sort_values(ascending = False), 
                       "C:\All\Dataset\Files\Correlation.csv")
sb.heatmap(correlation_matrix, annot = True) 
plt.xticks(rotation = 45)
plt.show()

train_set, test_set = train_test_split(dataframe_for_gold_close_prediction, test_size = 0.2, shuffle = False)
imputer = SimpleImputer(strategy = "median")
scaler = StandardScaler()
target_scaler = StandardScaler()

def get_dataframe_with_missing_values_filled_using_imputer(dataframe, imputer):
    dataframe_with_numerical_columns = dataframe.select_dtypes(include = [np.number])
    imputer.fit(dataframe_with_numerical_columns)
    dataframe_array = imputer.transform(dataframe_with_numerical_columns)
    return pd.DataFrame(dataframe_array, 
                        columns = dataframe_with_numerical_columns.columns, 
                        index = dataframe_with_numerical_columns.index)

def prep_data(dataframe, lookback, future, scaler, imputer, target_scaler):
    dataframe_lstm = dataframe[dataframe_columns_used]
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_lstm = get_dataframe_with_missing_values_filled_using_imputer(dataframe_lstm, imputer)
    dataframe_lstm_scaled = scaler.fit_transform(dataframe_lstm)
    
    dataframe_lstm_x = dataframe_lstm_scaled[:, 0 : number_of_features]
    dataframe_lstm_y = dataframe_lstm_scaled[:, number_of_features : ]
    
    x, y, dates_index = [], [], []
    for i in range(lookback, len(dataframe) - future + 1):
        x.append(dataframe_lstm_x[i - lookback : i, 0 : number_of_features])
        dates_index.append(i + future - 1)
        y.append(dataframe_lstm_y[i + future - 1 : i + future, 0])

    dates_train_lstm = pd.to_datetime(dataframe['Date'].iloc[dates_index[0] : dates_index[len(dates_index) - 1] + 1])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dates_train_lstm

Lstm_x, Lstm_y, train_dates = prep_data(dataframe = train_set, 
                                        lookback = 30, 
                                        future = 1, 
                                        scaler = scaler, 
                                        imputer = imputer,
                                        target_scaler = target_scaler)

def Lstm_model():
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (30, number_of_features)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mean_squared_error', metrics = ['mape'])
    return regressor

    #es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
    #regressor.fit(x, y, epochs = 100, batch_size = 64, verbose = 1, callbacks = [es])
    #return regressor
    
estimator = KerasRegressor(build_fn = Lstm_model, epochs = 100, batch_size = 50, verbose = 0)
early_stopping = EarlyStopping(monitor='loss', patience = 20, verbose = 1) 
history = estimator.fit(x = Lstm_x, 
                        y = Lstm_y, 
                        validation_split = 0.1,
                        epochs = 100, 
                        batch_size = 50, 
                        callbacks = [early_stopping], 
                        verbose = 1)


def visualize_learning_curve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mape'])
    plt.plot(history.history['val_mape'])
    plt.title('Mean Absolute Percentage Error')
    plt.ylabel('mape')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Mean Squared Error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
visualize_learning_curve(history)

def prep_test_data(dataframe, lookback, future, scaler, imputer):
    dataframe_lstm = dataframe[dataframe_columns_used]
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_lstm = get_dataframe_with_missing_values_filled_using_imputer(dataframe_lstm, imputer)
    dataframe_lstm_scaled = scaler.transform(dataframe_lstm)
    
    dataframe_lstm_x = dataframe_lstm_scaled[:, 0 : number_of_features]
    dataframe_lstm_y = dataframe[['Close']].astype(float)
    dataframe_lstm_y = dataframe_lstm_y.values
    
    x, y, dates_index = [], [], []
    for i in range(lookback, len(dataframe_lstm) - future + 1):
        x.append(dataframe_lstm_x[i - lookback : i, 0 : number_of_features])
        dates_index.append(i + future - 1)
        y.append(dataframe_lstm_y[i + future - 1 : i + future, 0])

    dates_lstm = pd.to_datetime(dataframe['Date'].iloc[dates_index[0] : dates_index[len(dates_index) - 1] + 1])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dates_lstm.values


Lstm_test_x, Lstm_test_y, test_dates = prep_test_data(dataframe = test_set,
                                                      lookback = 30,
                                                      future = 1,
                                                      scaler = scaler,
                                                      imputer = imputer)

predicted = estimator.predict(Lstm_test_x)#.reshape(-1, 1)
predicted_adjusted = np.repeat(predicted.reshape(-1, 1), number_of_features + 1, axis = -1)
predicted_descaled = scaler.inverse_transform(predicted_adjusted)[:, 0]
dataframe_for_plotting = pd.DataFrame({'Predicted' : predicted_descaled, 
                                       'Real' : Lstm_test_y[:, 0], 
                                       'Date' : test_dates})

def plot_dataframe(dataframe):
    dataframe.set_index('Date', inplace = True)
    plt.figure(figsize=(16,8))
    plt.title('Close Price Gold in USD')
    plt.ylabel('Close')
    plt.xlabel('Time')
    plt.plot(dataframe[['Real','Predicted']])
    plt.legend(['real', 'predicted'])
    plt.show()

plot_dataframe(dataframe_for_plotting)
rmse = mean_squared_error(pd.DataFrame({'Predicted' : predicted_descaled}).Predicted,
                          pd.DataFrame({'Real' : Lstm_test_y[:, 0]}).Real, 
                          squared = False)
print('rmse', rmse)
write_to_file('RMSE using LSTM' + str(rmse) + '\n', data_file_path)
mape = mean_absolute_percentage_error(pd.DataFrame({'Real' : Lstm_test_y[:, 0]}).Real, 
                                      pd.DataFrame({'Predicted' : predicted_descaled}).Predicted)
print('mape', mape)
write_to_file('MAPE using LSTM' + str(mape) + '\n', data_file_path)

def prep_data_rf(dataframe, lookback, future, scaler, imputer, target_scaler):
    dataframe_rf = dataframe[dataframe_columns_used]
    dataframe_rf = dataframe_rf.astype(float)
    dataframe_rf = get_dataframe_with_missing_values_filled_using_imputer(dataframe_rf, imputer)
    dataframe_rf_scaled = scaler.fit_transform(dataframe_rf)
    
    dataframe_rf_x = dataframe_rf_scaled[: len(dataframe_rf) - 1, 0 : number_of_features]
    dataframe_rf_y = dataframe_rf_scaled[1 : len(dataframe_rf), number_of_features :]

    dates_train_rf = pd.to_datetime(dataframe.iloc[1 : len(dataframe_rf), :]['Date'])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return dataframe_rf_x, dataframe_rf_y, dates_train_rf.values

def prep_test_data_rf(dataframe, lookback, future, scaler, imputer):
    dataframe_rf = dataframe[dataframe_columns_used]
    dataframe_rf = dataframe_rf.astype(float)
    dataframe_rf = get_dataframe_with_missing_values_filled_using_imputer(dataframe_rf, imputer)
    dataframe_rf_scaled = scaler.transform(dataframe_rf)
    
    dataframe_rf_x = dataframe_rf_scaled[: len(dataframe_rf) - 1, 0 : number_of_features]
    dataframe_rf_y = dataframe.iloc[1 : len(dataframe_rf), :].Close.values

    dates_rf = pd.to_datetime(dataframe.iloc[1 : len(dataframe_rf), :]['Date']).values
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return dataframe_rf_x, dataframe_rf_y, dates_rf


rf_x, rf_y, rf_train_dates = prep_data_rf(dataframe = train_set, 
                                          lookback = 30, 
                                          future = 1, 
                                          scaler = scaler, 
                                          imputer = imputer,
                                          target_scaler = target_scaler)

rf_test_x, rf_test_y, rf_test_dates = prep_test_data_rf(dataframe = test_set,
                                                        lookback = 30,
                                                        future = 1,
                                                        scaler = scaler,
                                                        imputer = imputer)


model_rf = RandomForestRegressor(n_estimators=500, 
                                 random_state=42, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 max_depth=10, 
                                 bootstrap=True)
model_rf.fit(rf_x, rf_y)
predicted_rf = model_rf.predict(rf_test_x)
predicted_rf_adjusted = np.repeat(predicted_rf.reshape(-1, 1), number_of_features + 1, axis = -1)
predicted_rf_descaled = scaler.inverse_transform(predicted_rf_adjusted)

def plot_rf_dataframe(dataframe):
    dataframe.set_index('Date', inplace = True)
    plt.figure(figsize=(16,8))
    plt.title('Close Price Gold in USD')
    plt.ylabel('Close')
    plt.xlabel('Time')
    plt.plot(dataframe[['RF_Real','RF_Predicted']])
    plt.legend(['real', 'predicted'])
    plt.show()

dataframe_rf_for_plotting = pd.DataFrame({'RF_Predicted' : predicted_rf_descaled[:,0], 
                                          'RF_Real' : rf_test_y[:], 
                                          'Date' : rf_test_dates})
plot_rf_dataframe(dataframe_rf_for_plotting)
rmse_rf = mean_squared_error(pd.DataFrame({'RF_Predicted' : predicted_rf_descaled[:,0]}).RF_Predicted,
                             pd.DataFrame({'RF_Real' : rf_test_y[:]}).RF_Real, 
                             squared = False)
print('rmse rf', rmse_rf)
write_to_file('RMSE using Random Forest Regression' + str(rmse_rf) + '\n', data_file_path)

mape_rf = mean_absolute_percentage_error(pd.DataFrame({'Real' : rf_test_y[:]}).Real, 
                                         pd.DataFrame({'Predicted' : predicted_rf_descaled[:,0]}).Predicted)
print('mape rf', mape_rf)
write_to_file('MAPE using Random Forest Regression' + str(mape_rf) + '\n', data_file_path)

ensemble_dataframe = pd.merge(dataframe_for_plotting, dataframe_rf_for_plotting, on='Date', how='inner')
ensemble_dataframe['Avg Prediction'] = abs(ensemble_dataframe['Predicted'] + ensemble_dataframe['RF_Predicted'])/2

avg_rmse = mean_squared_error(pd.DataFrame({'Real' : rf_test_y[29:]}).Real, ensemble_dataframe['Avg Prediction'], squared=False)
print('avg rmse', avg_rmse)
write_to_file('Average RMSE using forecast averaging' + str(avg_rmse) + '\n', data_file_path)

def plot_combined_dataframe(dataframe):
    dataframe.set_index('Date', inplace = True)
    plt.figure(figsize=(16,8))
    plt.title('Close Price Gold in USD')
    plt.ylabel('Close')
    plt.xlabel('Time')
    plt.plot(dataframe[['Real','Avg Predicted']])
    plt.legend(['real', 'predicted'])
    plt.show()

dataframe_combined_for_plotting = pd.DataFrame({'Avg Predicted' : ensemble_dataframe['Avg Prediction'], 
                                                'Real' : rf_test_y[29:], 
                                                'Date' : rf_test_dates[29:]})
plot_combined_dataframe(dataframe_combined_for_plotting)

def Lstm_baseline_model(dropout_rate = 0.2, optimizer = 'Adam', neurons = 50):
    regressor = Sequential()
    regressor.add(LSTM(units = neurons, return_sequences = True, input_shape = (30, number_of_features)))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(units = neurons, return_sequences = True))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(units = neurons, return_sequences = True))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(units = neurons))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units = 1))
    regressor.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics=['mae'])
    return regressor

es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
baseline_model = KerasRegressor(build_fn = Lstm_baseline_model, 
                                verbose = 1)

optimizer = ['adam']
dropout_rate = [0.2, 0.4]
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50]
batches = [10, 30, 50]
neurons = [50, 100]
param_grid = dict(dropout_rate = dropout_rate, 
                  epochs = epochs,
                  batch_size = batches,
                  optimizer = optimizer,
                  neurons = neurons)
    
grid = GridSearchCV(estimator = baseline_model, param_grid = param_grid, n_jobs = -1)
grid_result = grid.fit(Lstm_x, Lstm_y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))