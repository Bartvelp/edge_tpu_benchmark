import tensorflow as tf
import numpy as np
# https://www.tensorflow.org/lite/performance/post_training_quantization

def prepare_images():
	mnist = tf.keras.datasets.mnist
	(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
	# Convert the 8-bit numbers into floats between 0 and 1 as input 
	training_images, testing_images = training_images / 255.0, testing_images / 255.0

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
	# load model
	model = tf.keras.models.load_model('model.h5')

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	converter.representative_dataset = representative_dataset_gen
	tflite_quant_model = converter.convert()

	open("converted_model_from_keras_8bit_all.tflite", "wb").write(tflite_quant_model)
