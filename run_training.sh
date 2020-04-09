#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
export TF_CPP_MIN_LOG_LEVEL=2
python train_and_save_number_recog_NN.py $1
python perform_post_training_quant.py 
edgetpu_compiler converted_model_from_keras_8bit_all.tflite
scp converted_model_from_keras_8bit_all_edgetpu.tflite coral-dev:/home/mendel/thesis_scratchpad/converted_model_from_keras_8bit_all_edgetpu.tflite
scp converted_model_from_keras_8bit_all.tflite coral-dev:/home/mendel/thesis_scratchpad/converted_model_from_keras_8bit_all.tflite

echo "Run python edge_use_number_tflite.py on mendel"