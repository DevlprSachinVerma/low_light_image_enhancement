from structure import create_model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Compile the model
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.0009))

# Fit the model and capture the history
history = model.fit(
    train_low_images, train_high_images,
    batch_size=16,
    validation_data=val_dataset,
    epochs=150,
    verbose=1
)

# Save the model weights
model_save_path = '/content/drive/My Drive/model_weights4.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save_weights(model_save_path)
print(f'Model weights saved to {model_save_path}')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
