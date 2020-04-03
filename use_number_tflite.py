import numpy as np
import tensorflow as tf
import time

def current_milli_time(): # Helper time function
	return int(round(time.time() * 1000))

def prepare_images():
	mnist = tf.keras.datasets.mnist
	(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
	# Convert the 8-bit numbers into floats between 0 and 1 as input 
	training_images, testing_images = training_images / 255.0, testing_images / 255.0

	# convert it to a 32 bit float for tflite
	training_images, testing_images = training_images.astype(np.float32) , testing_images.astype(np.float32) 
	used_images = np.concatenate([training_images, testing_images])
	return used_images


def run_inference_round(interpreter, images):
	before = current_milli_time()

	for image in images:
		# print_greyscale(image)
		input_data = np.array([image]) # Needs to be wrapped for proper dimensions
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		# print('# {}, conf: {}'.format(np.argmax(output_data), np.max(output_data)))

	after = current_milli_time()
	return (before - after)

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
	interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
	interpreter.allocate_tensors()

	# Get input and output tensors shapes etc.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	images = prepare_images()
	for i in range(10):
		time_needed = run_inference_round(interpreter, images)
		print('{}: {} images took {} ms'.format(i, len(images), time_needed))
