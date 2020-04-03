import tensorflow as tf
mnist = tf.keras.datasets.mnist

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

# Convert the 8-bit numbers into floats between 0 and 1 as input 
training_images, testing_images = training_images / 255.0, testing_images / 255.0

model = tf.keras.models.Sequential([ # Sequential model, easy mindmap
    tf.keras.layers.Flatten(input_shape=(28, 28)), # we need to flatten the matrix into a vector 
    tf.keras.layers.Dense(128, activation='relu'), # Rectified linear activator
    tf.keras.layers.Dense(10), # 10 digits, so one_hot output
    tf.keras.layers.Softmax() # Softmax the previous output scores for the loss function
])

model.compile(
  optimizer='adam', # stochastic gradient descent method that just works well
  # easiest to use loss function sparse_categorical_crossentropy, easiest to understand mean_squared_error
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'] # Log accuracy
)

# Train for 5 runs trough the data, batch size is auto
model.fit(training_images, training_labels, epochs=5)
# Now the model is 98% accurate to the training data
model.summary()

# try it on the testing data
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
print('{} is the testing accuracy'.format(test_acc))

# Convert model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()
# Save tflite model to disk

open("converted_model_8bit.tflite", "wb").write(tflite_model)
