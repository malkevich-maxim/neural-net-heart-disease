import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report
from tensorflow.keras import regularizers

df = pd.read_csv('rawdata\heart_disease.csv')

input("ready?")

df['typical_angina_pain'] = df['cp'].replace([0, 1, 2, 3], [0, 1, 0, 0])
df['atypical_angina_pain'] = df['cp'].replace([0, 1, 2, 3], [0, 0, 1, 0])
df['non_anginal_pain'] = df['cp'].replace([0, 1, 2, 3], [0, 0, 0, 1])
df['no_pain'] = df['cp'].replace([0, 1, 2, 3], [1, 0, 0, 0])
df = df.drop(columns = 'cp')
df['restecg_normal'] = df['restecg'].replace([0, 1, 2], [0, 1, 0])
df['restecg_st_t_wave_abnormal'] = df['restecg'].replace([0, 1, 2], [0, 0, 1])
df['rest_ecg_hypertrophy'] = df['restecg'].replace([0, 1, 2], [1, 0, 0])
df = df.drop(columns = 'restecg')
df['peak_st_downslp'] = df['slp'].replace([0, 1, 2], [1, 0, 0])
df['peak_st_flatslp'] = df['slp'].replace([0, 1, 2], [0, 1, 0])
df['peak_st_upslp'] = df['slp'].replace([0, 1, 2], [0, 0, 1])
df = df.drop(columns = 'slp')
df = df.drop(df[df['thall'] == 0].index)
df['thal_fixed'] = df['thall'].replace([1, 2, 3], [1, 0, 0])
df['thal_normal'] = df['thall'].replace([1, 2, 3], [0, 1, 0])
df['thal_reversible'] = df['thall'].replace([1, 2, 3], [0, 0, 1])
df = df.drop(columns = 'thall')

X = df.drop(columns = 'output')
Y = df['output']
X = X.to_numpy()
Y = Y.to_numpy()
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.25,random_state=4)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
Y_train = Y_train.reshape(-1,1)
Y_val = Y_val.reshape(-1,1)

generated_results = pd.DataFrame({'model_id': [], 'layer_1': [], 'layer_2': [], 'layer_3': [], 'epochs': [], 'lambda': [], 'init_loss': [], 'init_val_loss': [], 'loss': [], 'val_loss': [], 'init_acc': [], 'init_val_acc': [], 'acc': [], 'val_acc': []})
la_list = []
for i in range(6):
  la_list.append(0.05 * i)
epoch_spacing = 10
epoch_list = []
for i in range(60):
  epoch_list.append(epoch_spacing * (i + 1))
relu_list = [0, 2, 5, 10]
model_id = -1
for la in la_list:
  for relu_1 in relu_list:
    for relu_2 in relu_list:
      if relu_1 == 0 and relu_2 != 0:
        break
      for relu_3 in relu_list:
        if relu_2 == 0 and relu_3 != 0:
          break
        model_id += 1
        model = 0
        model = Sequential()
        for i in [relu_1, relu_2, relu_3]:
          if i != 0:
            model.add(Dense(i, activation = 'relu', kernel_regularizer=regularizers.l2(la)))
        model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(la)))
        model.compile(loss = 'binary_crossentropy', metrics = ['binary_crossentropy', 'accuracy'])
        model.fit(X_train, Y_train, epochs = epoch_list[-1], verbose = 0, validation_data = (X_val, Y_val))
        model_history = pd.DataFrame(model.history.history)
        for epoch_amt in epoch_list:
          new = pd.DataFrame({'model_id': [model_id], 'layer_1': [relu_1], 'layer_2': [relu_2], 'layer_3': [relu_3], 'epochs': [epoch_amt], 'lambda': [la], 'init_loss': [model_history['loss'][0]], 'init_val_loss': [model_history['val_loss'][0]], 'loss': [model_history['loss'][epoch_amt - 1]], 'val_loss': [model_history['val_loss'][epoch_amt - 1]], 'init_acc': [model_history['accuracy'][0]], 'init_val_acc': [model_history['val_accuracy'][0]], 'acc': [model_history['accuracy'][epoch_amt - 1]], 'val_acc': [model_history['val_accuracy'][epoch_amt - 1]]})
          generated_results = pd.concat([generated_results, new], ignore_index=True)
        print(model_id)
generated_results.head()
