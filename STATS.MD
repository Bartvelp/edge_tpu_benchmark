# Speed stats compared to amount of parameters

| Parameters    | On-chip memory | off-chip memory | TPU (ms/500 img) | CPU (ms/500 img)|
| :------------ | :------------- | :-------------- | :--------------- | :-------------- |
| 20,587,010    | 7.82MiB        | 12.24MiB        | 16548            | **12751**       |
| 12,585,010    | 7.83MiB        | 4.54MiB         | **6611**         | 7837            |
| 7,999,550     | 7.84MiB        | 0               | **279**          | 5494            |
| 1,176,450     | 1.33MiB        | 0               | **198**          | 695             |
| 318,010       | 370KiB         | 0               | 156              | 154             |
| 15,910        | 51KiB          | 0               | 154              | **15**          |

If the TPU needs to stream to much parameters from off-chip memory it becomes slower than the onboard CPU

Then there is a sweet spot where the TPU can use only on-chip memory and hugely outperforms the CPU.

But with less parameters than ~318k the CPU performance keeps getting better, but the TPU is hitting a plateau of 155 ms / 500 images.

According to google the mobilenet v2 NN has 3.4 Million parameters, which puts it in the sweet spot of the TPU. The Mobilenet NN is the model used in the example by Coral of python classification [here](https://github.com/google-coral/tflite/blob/master/python/examples/classification/classify_image.py). 
## Method
Tested using the following format of a model, by varing the size of all the dense layers, and sometimes removing or adding some to get the wanted amount of params.
```
def create_model():
	startingLayerSize = 784
	sizeDenseLayers = 200
	model = tf.keras.models.Sequential([ # Sequential model, easy mindmap
			tf.keras.layers.Dense(startingLayerSize, activation='relu', input_shape=(784,)), # Rectified linear activator
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(10, activation='relu'),
			tf.keras.layers.Softmax()	# Softmax the previous output scores for the loss function
	])
	model.compile(
		optimizer='sgd', # stochastic gradient descent method that just works well
		# easiest to use loss function sparse_categorical_crossentropy, easiest to understand mean_squared_error
		loss='mean_squared_error',
		metrics=['accuracy'] # Log accuracy
	)
	return model
```
