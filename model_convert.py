import tensorflow as tf
import tensorflowjs as tfjs

# Load your new .keras model
model = tf.keras.models.load_model("gesture_model.keras")

# Convert to TensorFlow.js
tfjs.converters.save_keras_model(model, "tfjs_model")