import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import time
from sys import argv
from os import listdir

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
        input_data = np.array([image])    # Needs to be wrapped for proper dimensions
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        inference_times.append(inference_time)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print('# {}, conf: {}'.format(np.argmax(output_data), np.max(output_data)))
    return sum(inference_times) * 1000

def get_interpreter (isEdgeTPU, model_path):
    if isEdgeTPU:
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        return interpreter
    else:
        return tflite.Interpreter(model_path=model_path)

if __name__ == "__main__":
    tflite_fns = sorted(['./models/' + fn for fn in listdir('./models') if fn.endswith('.tflite')])
    edge_tflite_fns = sorted(['./edgetpu_models/' + fn for fn in listdir('./edgetpu_models') if fn.endswith('.tflite')])

    images = prepare_images()
    
    print("Running on TPU")
    for edge_tflite_fn in edge_tflite_fns:
        # Load TFLite model and allocate tensors.
        TPU_interpreter = get_interpreter(True, edge_tflite_fn)
        TPU_interpreter.allocate_tensors()
        times = []
        for i in range(3):
            time_needed = run_inference_round(TPU_interpreter, images)
            times.append(time_needed)
        avg_time = sum(times) / len(times)
        print('{} images took {:.2f} ms TPU {}'.format(
            len(images), avg_time, edge_tflite_fn))

    print("Running on CPU")
    for tflite_fn in tflite_fns:
        # Load TFLite model and allocate tensors.
        CPU_interpreter = get_interpreter(False, tflite_fn)
        CPU_interpreter.allocate_tensors()
        times = []
        for i in range(3):
            time_needed = run_inference_round(CPU_interpreter, images)
            times.append(time_needed)
        avg_time = sum(times) / len(times)
        print('{} images took {:.2f} ms CPU {}'.format(len(images), avg_time, tflite_fn))   
    print('Done')
