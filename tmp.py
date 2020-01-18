def make_train_model(self):
    input_data = kl.Input(shape=(1, self.input_shape))
    lstm = kl.LSTM(5, input_shape=(1, self.input_shape), return_sequences=True,activity_regularizer=regularizers.l2(0.003),
        recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
    perc = kl.Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
    lstm2 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
        dropout=0.2, recurrent_dropout=0.2)(perc)
    out = kl.Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)


model = Model(input_data, out)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

### from 01