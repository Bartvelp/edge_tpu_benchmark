import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import time
from sys import argv

def current_milli_time(): # Helper time function
	return int(round(time.time() * 1000))

def prepare_images():
	(training_images, training_labels), (testing_images, testing_labels) = pickle.load(open('mnist_data.pickle', 'rb'))
	used_images = np.concatenate([training_images, testing_images])
	return used_images[0:5000]


def run_inference_round(interpreter, images):
# Get input and output tensors shapes etc.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	before = current_milli_time()
	for image in images:
		# print_greyscale(image)
		input_data = np.array([image]) # Needs to be wrapped for proper dimensions
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		# print('# {}, conf: {}'.format(np.argmax(output_data), np.max(output_data)))

	after = current_milli_time()
	return (after - before)

def get_interpreter (isEdgeTPU):
	if isEdgeTPU:
		interpreter = tflite.Interpreter(
			model_path="converted_model_from_keras_8bit_all_edgetpu.tflite", 
			experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
		)
		return interpreter
	else:
		return tflite.Interpreter(model_path="converted_model_from_keras_8bit_all.tflite")

if __name__ == "__main__":
	# Currently takes
	# 1876 ms om the Edge TPU
	# 700 ms on the CPU ...

	images = prepare_images()
	# Load TFLite model and allocate tensors.
	CPU_interpreter = get_interpreter(False)
	TPU_interpreter = get_interpreter(True)
	CPU_interpreter.allocate_tensors()
	TPU_interpreter.allocate_tensors()
	print('Starting inference rounds on CPU')
	for i in range(5):
		time_needed = run_inference_round(CPU_interpreter, images)
		print('{}: {} images took {} ms'.format(i, len(images), time_needed))
	
	print('Starting inference rounds on TPU')
	for i in range(5):
		time_needed = run_inference_round(TPU_interpreter, images)
		print('{}: {} images took {} ms'.format(i, len(images), time_needed))
