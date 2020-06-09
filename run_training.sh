#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
export TF_CPP_MIN_LOG_LEVEL=2
python train_and_save_number_recog_NN.py
python perform_post_training_quant.py 
python convert_to_edgeTPU.py
if [ "$(whoami)" == "bart" ] ; then
    echo "scp ./models/*.tflite coral-dev:/home/mendel/thesis_scratchpad/models/"
    echo "scp ./edgetpu_models/* coral-dev:/home/mendel/thesis_scratchpad/edgetpu_models/"
else
    echo "Please copy the models in edgetpu_models and models dirs to the coral dev board"
fi

echo "Run python edge_use_number_tflite.py on mendel"