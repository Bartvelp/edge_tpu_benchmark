import tensorflow as tf

import pickle

def get_images():
  mnist = tf.keras.datasets.mnist
  (training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
  # Convert the 8-bit numbers into floats between 0 and 1 as input 
  training_images, testing_images = training_images / 255.0, testing_images / 255.0
  return training_images, testing_images, training_labels, testing_labels

def create_model():
  model = tf.keras.models.Sequential([ # Sequential model, easy mindmap
      tf.keras.layers.Flatten(input_shape=(28, 28)), # we need to flatten the matrix into a vector 
      tf.keras.layers.Dense(50, activation='relu'), # Rectified linear activator
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Softmax()  # Softmax the previous output scores for the loss function
  ])
  model.compile(
    optimizer='sgd', # stochastic gradient descent method that just works well
    # easiest to use loss function sparse_categorical_crossentropy, easiest to understand mean_squared_error
    loss='mean_squared_error',
    metrics=['accuracy'] # Log accuracy
  )
  return model

if __name__ == '__main__':
  training_images, testing_images, training_labels, testing_labels = get_images()
  model = create_model()
  one_hot_training_labels = tf.keras.utils.to_categorical(training_labels)
  one_hot_testing_labels = tf.keras.utils.to_categorical(testing_labels)
  # Train for 5 runs trough the data, batch size is auto
  model.fit(training_images, one_hot_training_labels, epochs=1)
  # Now the model is 98% accurate to the training data
  model.summary()
  #tf.keras.utils.plot_model(model, to_file='model.png')
  
  # try it on the testing data
  test_loss, test_acc = model.evaluate(testing_images, one_hot_testing_labels, verbose=2)
  print('{} is the testing accuracy'.format(test_acc))
  # Save model in h5 format to disk
  model.save("model.h5")
  # also save the MNIST data to disk for inference testing
  pickle.dump(tf.keras.datasets.mnist.load_data(), open('mnist_data.pickle', 'wb'))
  print('Saved model and MNIST data')
