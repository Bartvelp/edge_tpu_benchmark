import tensorflow as tf
import numpy as np
import pickle
from sys import argv

def get_images():
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
    # Convert the 8-bit numbers into floats between 0 and 1 as input 
    training_images, testing_images = training_images / 255.0, testing_images / 255.0
    training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], 28, 28, 1)

    return training_images[:5000], testing_images[:20], training_labels[:5000], testing_labels[:20]


def create_model(typeNN):
    num_layers = 100
    model = tf.keras.models.Sequential() # Sequential model, easy mindmap

    if typeNN == 'dense':
        size_layers = 50
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1))) # we need to flatten the matrix into a vector
        for i in range(num_layers):
            model.add(tf.keras.layers.Dense(size_layers))
        
    elif typeNN == 'conv':
        num_filters = 32
        kernel_size = 8
        model.add(tf.keras.layers.Conv2D(num_filters, kernel_size, input_shape=(28, 28, 1), padding='same'))
        for i in range(num_layers):
            model.add(tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same'))
        # Flatten afterwards
        model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # For the loss function
    
    model.compile(
        optimizer='adam', # stochastic gradient descent method that just works well
        # easiest to use loss function sparse_categorical_crossentropy, easiest to understand mean_squared_error
        loss='mean_squared_error',
        metrics=['accuracy'] # Log accuracy
    )
    return model


if __name__ == '__main__':
    typeNN = argv[1] # 'dense' or 'conv'
    print('STARTING to learn NN of type: ' + typeNN)
    training_images, testing_images, training_labels, testing_labels = get_images()
    model = create_model(typeNN)
    one_hot_training_labels = tf.keras.utils.to_categorical(training_labels)
    one_hot_testing_labels = tf.keras.utils.to_categorical(testing_labels)
    # Train for 5 runs trough the data, batch size is auto
    # model.fit(training_images, one_hot_training_labels, epochs=1)
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
