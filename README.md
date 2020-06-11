## Installation
```
tensorflow==2.1.0
```
See requirements.txt for the full list.
And python 3 (v3.7.3) and the [edgetpu_compiler](https://coral.ai/docs/edgetpu/compiler/#system-requirements) must be installed on the pc
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