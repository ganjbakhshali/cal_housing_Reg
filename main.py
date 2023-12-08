from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()
x_full, x_test, y_full, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation=keras.activations.relu,
#                        input_shape=x_train.shape[1:]),
#     keras.layers.Dense(1)
# ])

# model2

input_ = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation=keras.activations.relu)(input_)
dropout1 = keras.layers.Dropout(0.5)(hidden1)  # Add dropout layer
hidden2 = keras.layers.Dense(30, activation=keras.activations.relu)(dropout1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model2 = keras.Model(inputs=[input_], outputs=[output])

model2.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mse'])


# model.compile(loss=keras.losses.mean_squared_error,
#               optimizer=keras.optimizers.SGD(), metrics=['mse'])


# history = model.fit(x_train, y_train, epochs=20,
#                     validation_data=(x_val, y_val))

history2 = model2.fit(x_train, y_train, epochs=20,
                     validation_data=(x_val, y_val), batch_size=32)


pd.DataFrame(history2.history).plot(figsize=(10, 6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

mse_test = model2.evaluate(x_test, y_test)
x_new = x_train[:3]
y_pred = model2.predict(x_new)
