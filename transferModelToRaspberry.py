import tensorflow as tf
import os
import raspberryFileTransfer

# Model path
modelsFolder = 'Models'
modelName = 'testModel'

# Raspberry Pi connection details
hostname = '192.168.236.125'
username = 'pi'
password = 'raspberry'
remoteProjectFolder = os.path.join('Projects', 'EngineeringProject')

# Load your trained model
model = tf.keras.models.load_model(os.path.join(modelsFolder, modelName + '.keras'))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open(os.path.join(modelsFolder, modelName + '.tflite'), 'wb') as file:
    file.write(tflite_model)

raspberryFileTransfer.transfer_file(
    local_path=os.path.join(modelsFolder, modelName + ".tflite"),
    remote_path=os.path.join(remoteProjectFolder, modelsFolder, modelName + ".tflite"),
    hostname=hostname,
    username=username,
    password=password,
    port=22  # Default SSH port, change if needed
)
