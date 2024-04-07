import tensorflow as tf
import os

# Model path
modelsFolder = 'Models'
modelName = 'testModel'


# Define a simple sequential model
def create_model():
    # Create a simple model with a single dense layer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create and save the model
if __name__ == "__main__":
    # Create a simple model
    model = create_model()

    # Optionally, you can train the model here with your data
    # For example: model.fit(x_train, y_train, epochs=10)

    # Create folder if it does not exist
    if not os.path.exists(modelsFolder):
        os.makedirs(modelsFolder)

    # Save the model in the TensorFlow SavedModel format
    model.save(os.path.join(modelsFolder, modelName + '.keras'))

    # If you want to save the model in the HDF5 format, use:
    # model.save('path/to/save/your/model.h5')
