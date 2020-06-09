import subprocess
from os import listdir


if __name__ == '__main__':
  tflite_fns = sorted(['./models/' + fn for fn in listdir('./models') if fn.endswith('.tflite')])
  for tflite_fn in tflite_fns:
    print('Converting: ' + tflite_fn)
    subprocess.run(['edgetpu_compiler', '-o', './edgetpu_models/', tflite_fn])
  print('Done')
