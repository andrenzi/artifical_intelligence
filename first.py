import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsium_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farengete_q = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

print(celsium_q, farengete_q)

for a, b in enumerate(celsium_q):
    print(celsium_q[a])

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

history = model.fit(celsium_q, farengete_q, epochs = 10000, verbose=False)
print('complete the learning')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()

print(model.predict([55.0]))
print('This is a varables of area: {}'.format(l0.get_weights()))
