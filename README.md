## Packages
```
tensorflow==2.1.0
numpy==1.18.2
```
And python 3 (v3.7.3) and the edgetpu_compiler
## Order to run files
### PC
Make sure you create empty `models` and `edgetpu_models` folders first.
```
./run_training.sh
```

### Coral Dev board
```
python edge_use_number_tflite.py
```