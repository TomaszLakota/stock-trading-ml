import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras.models import Model
np.random.seed(4)
from tensorflow import set_random_seed
set_random_seed(4)
from util import csv_to_dataset, history_points

###set my settings
LOAD_MODEL_FROM_FILE = False



# dataset
# ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('my.csv')
ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('my.csv')

test_split = 0.7
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)
print(ohlcv_histories.shape)
print(ohlcv_histories[0][0])

######################################################

if LOAD_MODEL_FROM_FILE:
    model = load_model('model.h5')
    model.summary()
else:
    lstm_input = Input(shape=(history_points, 2), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    # x = LSTM(50, name='lstm_0')(lstm_input)
    # x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# TB = TensorBoard(histogram_freq=1, batch_size=32)
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=20, shuffle=True, validation_split=0.2, verbose=1)

###################################################### evaluation

y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)



plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

model.save("modela.h5")
print("saved")
