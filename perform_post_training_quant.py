import tensorflow as tf
import numpy as np
# https://www.tensorflow.org/lite/performance/post_training_quantization
# if you get a toco_from_protos: not found error
# Add ~/.local/bin/ to your PATH. e.g: PATH=/home/bart/.local/bin/:$PATH
def prepare_images():
	mnist = tf.keras.datasets.mnist
	(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
	# Convert the 8-bit numbers into floats between 0 and 1 as input 
	training_images, testing_images = training_images / 255.0, testing_images / 255.0
	# Flatten images here
	training_images = training_images.reshape(training_images.shape[0], 28 * 28)
	testing_images = testing_images.reshape(testing_images.shape[0], 28 * 28)

	# convert it to a 32 bit float for tflite
	training_images, testing_images = training_images.astype(np.float32) , testing_images.astype(np.float32) 
	used_images = np.concatenate([training_images, testing_images])
	return used_images


images = prepare_images()

def representative_dataset_gen():
	for i in range(1, 10000):
		# Get sample input data as a numpy array in a method of your choosing.
		yield [images[i:i+1]]

if __name__ == '__main__':
	# load model, compat mode because https://github.com/google-coral/edgetpu/issues/13
	converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('model.h5')
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	converter.representative_dataset = representative_dataset_gen
	tflite_quant_model = converter.convert()

	filename = 'converted_model_from_keras_8bit_all.tflite'
	open(filename, 'wb').write(tflite_quant_model)
	print('Saved model with 8bit quanitization')
	print('Run the following command to compile it for the edge tpu:')
	print('edgetpu_compiler {}'.format(filename))
