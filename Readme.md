# Introduction

This code is based on [https://github.com/lizhihao6/Forward-Warp](https://github.com/lizhihao6/Forward-Warp) and modified for the project: `On the RCS Estimation from Camera Images`.




## Foward Warp Pytorch Version

Has been tested in pytorch=0.4.0, python=3.6, CUDA=9.0


### Install

## Linux
Execute
```bash
export CUDA_HOME=/usr/local/cuda #use your CUDA instead
chmod a+x install.sh
./install.sh
```

## Windows
Execute
```bash
install.bat
```


### Test

```bash
cd test
python test.py
```

### Usage

```python
from Forward_Warp import forward_warp

# default interpolation mode is Bilinear
fw = forward_warp()
im_pred = torch.zeros_like(flow)
im_pred = fw(im0, flow, im_pred) 
```
