## Packages
```
tensorflow==2.1.0
numpy==1.18.2
```
And python 3 (v3.7.3) and the edgetpu_compiler
## Order to run files
### PC
```
python train_and_save_number_recog_NN.py
python perform_post_training_quant.py
edgetpu_compiler converted_model_from_keras_8bit_all.tflite
```

### Coral Dev board
```
python edge_use_number_tflite.py
```