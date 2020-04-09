import numpy as np
import tensorflow as tf
import pickle
import time

def prepare_images():
    (training_images, training_labels), (testing_images, testing_labels) = pickle.load(open('mnist_data.pickle', 'rb'))

    training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], 28, 28, 1)
    used_images = np.concatenate([training_images, testing_images])
    return used_images[:100]


def run_inference_round(interpreter, images):
    inference_times = []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for image in images:
        # print_greyscale(image)
        input_data = np.array([image]) # Needs to be wrapped for proper dimensions
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        inference_times.append(inference_time)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print('# {}, conf: {}'.format(np.argmax(output_data), np.max(output_data)))
    return sum(inference_times) * 1000

def print_greyscale(pixels, width=28, height=28):
    # helper print function adopted from https://stackoverflow.com/a/44052237/5329317
    pixels = pixels.flatten()
    def get_single_greyscale(pixel):
        val = 232 + round(pixel * 23)
        return '\x1b[48;5;{}m \x1b[0m'.format(int(val))
    for l in range(height):
        line_pixels = pixels[l * width:(l+1) * width]
        print(''.join(get_single_greyscale(p) for p in line_pixels))

if __name__ == "__main__":
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_model_from_keras_8bit_all.tflite")
    interpreter.allocate_tensors()
    print('Starting inference on desktop CPU')
    images = prepare_images()
    for i in range(5):
        time_needed = run_inference_round(interpreter, images)
        print('{}: {} images took {} ms'.format(i, len(images), time_needed))
