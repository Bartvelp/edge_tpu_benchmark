#!/bin/bash
python train_and_save_number_recog_NN.py 
python perform_post_training_quant.py 
edgetpu_compiler converted_model_from_keras_8bit_all.tflite
scp converted_model_from_keras_8bit_all_edgetpu.tflite coral-dev:/home/mendel/thesis_scratchpad/converted_model_from_keras_8bit_all_edgetpu.tflite
scp converted_model_from_keras_8bit_all.tflite coral-dev:/home/mendel/thesis_scratchpad/converted_model_from_keras_8bit_all.tflite

echo "Run python edge_use_number_tflite.py on mendel"