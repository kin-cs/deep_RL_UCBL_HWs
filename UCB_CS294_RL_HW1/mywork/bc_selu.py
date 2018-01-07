import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.noise import AlphaDropout
from keras import optimizers

import pickle

from sklearn.preprocessing import MinMaxScaler

batch_size = 1000
epochs = 500
plot = True


n_dense = 5
dense_units = 40
activation = 'selu'
#activation = 'relu'
dropout = AlphaDropout
#dropout = Dropout
#dropout_rate = 0.2
dropout_rate = 0.0
kernel_initializer = 'lecun_normal'
#kernel_initializer = 'glorot_uniform'
#optimizer = 'sgd'
#optimizer = optimizers.SGD(lr=0.1,decay=0.0001)
optimizer = optimizers.Adam(lr=0.001)
in_dim = 111
out_dim = 8

model = Sequential()
model.add(Dense(dense_units, input_shape=(in_dim,),
                kernel_initializer=kernel_initializer))
model.add(Activation(activation))
model.add(dropout(dropout_rate))

for i in range(n_dense - 1):
    model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

model.add(Dense(out_dim))
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

expert_data = pickle.load(open('../expert_data.pkl','rb'))
x_train = expert_data['observations']
y_train = expert_data['actions'][:,0,:]

xScaler = MinMaxScaler()
x_train = xScaler.fit_transform(x_train)
yScaler = MinMaxScaler()
y_train = yScaler.fit_transform(y_train)

scalers = {
           'xScaler':xScaler,
           'yScaler':yScaler,
           }
pickle.dump(scalers,open('scalers.pkl','wb'))

checkpoint_cb = keras.callbacks.ModelCheckpoint('trained_selu_model.pkl', monitor='val_loss')
model.fit(x_train,
           y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_split=0.1,
           callbacks = [checkpoint_cb])

