# Stats

## Best case scenerio
### Biggest time saver
7,999,550 params equals to 7.84MiB On-chip memory used for caching model parameters
CPU 500 images took 5494 ms
TPU 500 images took 279 ms

## In between
1,176,450 params 1.33MiB on chip memory
CPU 500 images took 695 ms
TPU 500 images took 198 ms

## Break even
318,010 params 370.00KiB on chip
CPU 500 images took 154 ms
TPU 500 images took 156 ms


# With less parameters than ~318k the CPU performance keeps getting better, but the TPU is hitting a plateau of 155 ms / 500 images.

# Method
Tested using the following model, by varing the size of all the dense layers
```
def create_model():
	sizeDenseLayers = 200
	model = tf.keras.models.Sequential([ # Sequential model, easy mindmap
      tf.keras.layers.Dense(784, activation='relu', input_shape=(784,)), # Rectified linear activator
      tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
      tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
			tf.keras.layers.Dense(sizeDenseLayers, activation='relu'),
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

For the break even I resorted to this model:

```
def create_model():
	model = tf.keras.models.Sequential([ # Sequential model, easy mindmap
			tf.keras.layers.Dense(400, input_shape=(784,)), # Rectified linear activator
			tf.keras.layers.Dense(10, activation='relu'), # 10 digits, so one_hot output
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