import tensorflow as tf
import numpy as np

x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

# Proveravamo da je y u dobrom obliku
if len(y.shape) == 1:
    num_classes = len(np.unique(y))  # brojimo jedinstvene akcije
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

# Definisemo model neuronske mreze
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=x.shape[1:]),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])

# Kompajlira se model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treniramo model
model.fit(x, y, epochs=100, batch_size=64, validation_split=0.2)

model.save('trained_model.keras')

