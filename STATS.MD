# Speed stats compared to amount of parameters

## Convultional
| Parameters    | On-chip memory | off-chip memory | TPU (ms/100 img) | CPU (ms/100 img)|
| :------------ | :------------- | :-------------- | :--------------- | :-------------- |
| 2,586,490     | 6.78MiB        | 892.12KiB       | 770.81           | 60871.17        |
| 946,490       | 3.18MiB        | 196.06KiB       | 326.31           | 20364.11        |
| 290,490       | 664KiB         | 196.06KiB       | 162.82           | 4174.98         |
| 208,490       | 340KiB         | 196.06KiB       | 133.46           | 2150.38         |

## Dense
| Parameters    | On-chip memory | off-chip memory | TPU (ms/100 img) | CPU (ms/100 img)|
| :------------ | :------------- | :-------------- | :--------------- | :-------------- |
| 10,168,810    | 7.94MiB        | 5.23MiB         | 1762.97          | 1184.59         |
| 6,550,810     | 7.94MiB        | 768.19KiB       | 326.31           | 768.22          |
| 5,746,810     | 7.70MiB        | 0               | 201.59           | 673.80          |
| 520,810       | 1.23MiB        | 0               | 52.48            | 51.45           |

If the TPU needs to stream to much parameters from off-chip memory it becomes slower than the onboard CPU
