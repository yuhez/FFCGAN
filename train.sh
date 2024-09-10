#!/bin/bash
EXP='run99_camera2_tv_std_augment_mse100'
LOG=logs/$EXP.log
nohup python3 FFCGAN.py > $LOG 2>&1 &
echo Detached