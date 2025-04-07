# -*- coding: utf-8 -*-
"""
Paper: Using neural networks as an alternative to air dispersion modeling in environmental impact assessment.

Authors: Mateo Concha and Gonzalo Ruz.

Description: Code used for the paper development.

"""

## Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

"""
    Metric functions
"""

def root_mean_squared_error(y_true, y_pred):
    rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred-y_true)))
    return rmse.numpy()

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

"""
    Datasets loading and pre-processing
"""

## Training, validation, and tests datasets
train_data = pd.read_csv('train.csv')
validate_data = pd.read_csv('Validation.csv')
test_FEEE_data = pd.read_csv("test_FFEE.csv")
test_CV_data = pd.read_csv("test_CV.csv")

train_data, _ = train_test_split(train_data, 
                                 test_size=0.3, 
                                 random_state=123)
print('(shape - train) {}'.format(train_data.shape))
print('(shape - validate) {}'.format(validate_data.shape))
print('(shape - test_FFEE) {}'.format(test_FEEE_data.shape))
print('(shape - test_CV) {}'.format(test_CV_data.shape))

## Separate the datasets into predictors and targets (SO2 Concentration)
X_train, y_train = train_data.drop('FFEE', axis=1), train_data['FFEE']
X_val, y_val = validate_data.drop('FFEE', axis=1), validate_data['FFEE']
X_test_FFEE, y_test_FFEE = test_FEEE_data.drop('FFEE', axis=1), test_FEEE_data['FFEE']
X_test_CV, y_test_CV = test_CV_data.drop('Chacaya', axis=1), test_CV_data['Chacaya']

## Data Scaling
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val) 
X_test_FFEE = scaler.transform(X_test_FFEE)
X_test_CV = scaler.transform(X_test_CV)

"""
    Model builder and training
"""

## Neural network builder
model = Sequential([
    Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

## Model compiler setting
model.compile(optimizer=RMSprop(), loss='mean_squared_error')

## Model training
history=model.fit(X_train, y_train, 
                  epochs=200, batch_size=256, 
                  validation_split=0.3
                  )

## Performance display
plt.plot(history.history['loss'],label='RMSE train')
plt.plot(history.history['val_loss'],label='RMSE val')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

"""
    Model evaluation on FFEE validation dataset
"""

## Compute prediction
prediction_val = model.predict(X_val)
comparison_val_FFEE = pd.DataFrame({
    'Ground_true': y_val.values,
    'Predicted': prediction_val.flatten()
})
comparison_val_FFEE

## FFEE ground truth and prediction visualization
plt.figure(figsize=(12, 6))
plt.plot(y_val, label='FFEE')
plt.plot(prediction_val, label='NN', linestyle='--')
plt.title('Ground Truth and Predictions')
plt.xlabel('Time')
plt.ylabel(r'Concentration SO$_2$ (ug/$m^3$)')
plt.legend()
plt.show()

## RMSE and MAE computation
a = root_mean_squared_error(np.array(y_val), np.array(prediction_val))
b = mae(y_test_FFEE, prediction_val)
print('RMSE: {}, MAE: {}'.format(a, b))

"""
    Model evaluation on FFEE test dataset
"""

## Compute prediction
prediction_FFEE = model.predict(X_test_FFEE)
comparison_test_FFEE = pd.DataFrame({
    'Ground_true': y_test_FFEE.values,
    'Predicted': prediction_FFEE.flatten()
})
comparison_test_FFEE

## FFEE ground truth and prediction visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_FFEE, label='FFEE')
plt.plot(prediction_FFEE, label='NN', linestyle='--')
plt.title('Ground Truth and Predictions')
plt.xlabel('Time')
plt.ylabel(r'Concentration SO$_2$ (ug/$m^3$)')
plt.legend()
plt.show()

## RMSE and MAE computation
a = root_mean_squared_error(np.array(y_test_FFEE), np.array(prediction_FFEE))
b = mae(y_test_FFEE, prediction_FFEE)
print('RMSE: {}, MAE: {}'.format(a, b))


np.savetxt("predictions_FFEE_20_10_200_256.csv", prediction_FFEE, delimiter=",")

"""
    Model evaluation on CV test dataset
"""

## Compute prediction
prediction_CV = model.predict(X_test_CV)
comparison_test_CV= pd.DataFrame({
    'Ground_true': y_test_CV.values,
    'Predicted': prediction_CV.flatten()
})
comparison_test_CV

## CV ground truth and prediction visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_CV, label='CV')
plt.plot(prediction_CV, label='NN', linestyle='--')
plt.title('Ground Truth and Predictions')
plt.xlabel('Time')
plt.ylabel(r'Concentration SO$_2$ (ug/$m^3$)')
plt.legend()
plt.show()
 
## RMSE and MAE computation
a = root_mean_squared_error(np.array(y_test_CV), np.array(prediction_CV))
b = mae(y_test_CV, prediction_CV)
print('RMSE: {}, MAE: {}'.format(a, b))

np.savetxt("predictions_CV.csv", prediction_CV, delimiter=",")

##############################################################################################