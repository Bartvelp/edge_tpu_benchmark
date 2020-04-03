import tensorflow as tf
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

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = prepare_images()
tflite_quant_model = converter.convert()
