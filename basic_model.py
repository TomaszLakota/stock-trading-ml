import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, BatchNormalization
from keras.models import Model
from util import csv_to_dataset, history_points

###set my settings
LOAD_MODEL_FROM_FILE = False
MODEL_LOAD_NAME = "tmp.h5"
MODEL_SAVE_NAME = "tmp1.h5"
EPOCHS = 1



# dataset
ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('my.csv')


prediction = 111
ohlcv_histories = ohlcv_histories[:-prediction]
next_day_open_values = next_day_open_values[prediction:]

test_split = 0.7
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y = unscaled_y[:-prediction]
unscaled_y_test = unscaled_y[n:]

# print(ohlcv_train.shape)
# print(ohlcv_test.shape)
# print(ohlcv_histories.shape)
# print(ohlcv_histories[0][0])

######################################################

if LOAD_MODEL_FROM_FILE:
    model = load_model(MODEL_LOAD_NAME)
else:
    lstm_input = Input(shape=(history_points, 4), name='lstm_input')

    x = LSTM(50, name='lstm_0',return_sequences=False)(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    # x = LSTM(50, name='lstm_1')(x)
    # x = Dropout(0.2, name='lstm_dropout_1')(x)
    x = Dense(50, name='dense_0')(x)
    x = Activation('relu', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

model.summary()
adam = optimizers.Adam(lr=0.0005) # 0005
model.compile(optimizer=adam, loss='mse')

filepath = "weights/"+MODEL_LOAD_NAME+"weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
TB = TensorBoard(histogram_freq=1, batch_size=32)
callbacks_list = [checkpoint, TB]

model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=EPOCHS, shuffle=True, validation_split=0.2, verbose=0,callbacks=callbacks_list)

###################################################### evaluation

y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

print(unscaled_y_test.shape)
print(y_test_predicted.shape)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)
print(real_mse)

model.save(MODEL_SAVE_NAME)
print("saved")


# display graphs
start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.gcf().set_size_inches(11, 5, forward=True)
plt.legend(['Real', 'Predicted'])
plt.show()


real = plt.plot(unscaled_y_test[-222:-1], label='real')
pred = plt.plot(y_test_predicted[-222:-1], label='predicted')
plt.show()