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

    return training_images[:10], testing_images[:10], training_labels[:10], testing_labels[:10]


def create_model(typeNN, num_layers):
    model = tf.keras.models.Sequential() # Sequential model, easy mindmap

    if typeNN == 'dense':
        size_layers = 200
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1))) # we need to flatten the matrix into a vector
        for i in range(num_layers):
            model.add(tf.keras.layers.Dense(size_layers))
        
    elif typeNN == 'conv':
        num_filters = 16
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
    NN_types = ['dense', 'conv']
    NN_num_layers_dense = [10, 140, 160, 250]
    NN_num_layers_conv = [5, 10, 50, 150] # has a lot more parameters per layer
    training_images, testing_images, training_labels, testing_labels = get_images()
    one_hot_training_labels = tf.keras.utils.to_categorical(training_labels)
    one_hot_testing_labels = tf.keras.utils.to_categorical(testing_labels)
    # also save the MNIST data to disk for inference testing
    pickle.dump(tf.keras.datasets.mnist.load_data(),
                open('mnist_data.pickle', 'wb'))

    for typeNN in NN_types:
        if typeNN == 'dense':
            num_layers_arr = NN_num_layers_dense
        elif typeNN == 'conv':
            num_layers_arr = NN_num_layers_conv
        for num_layers in num_layers_arr:
            print('STARTING to learn NN of type: {}, num layers: {}'.format(typeNN, num_layers))
            model = create_model(typeNN, num_layers)
            # Train for 1 runs trough the data (only 10 images), batch size is auto
            if typeNN is 'dense':
                model.fit(training_images, one_hot_training_labels, epochs=1)

            model_fn = 'models/{}_{}params.h5'.format(typeNN, model.count_params())
            # Save model in h5 format to disk
            model.save(model_fn)
            print('Saved model: ' + model_fn)
