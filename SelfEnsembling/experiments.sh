#!/usr/bin/env bash

#  ----------------  SVHN TO MNIST    ---------------------  #

# SOURCE ONLY
screen python3 main_toso.py --total_epochs 30 --learning_rate 0.001 --truncate_mnist False --source svhn --target mnist --channel_size 3 --model_dir ./model_s2m/run_1_so
# TARGET ONLY
screen python3 main_toso.py --truncate_mnist False --total_epochs 20 --learning_rate 0.01 --source mnist --target svhn --channel_size 3 --model_dir ./model_s2m/run_1_taso
# ADAPTATION
screen python3 main.py --truncate_mnist False --total_epochs 100 --rampup_epochs 80 --learning_rate 0.01 --source svhn --target mnist --channel_size 3 --model_dir ./model_s2m/run_1


#  ----------------  MNIST TO MNIST-M ---------------------  #

# SOURCE ONLY
screen python3 main_toso.py --total_epochs 30 --learning_rate 0.001 --source mnist --target mnistm --channel_size 1 --model_dir ./model_m2mm/run_1_so
# TARGET ONLY
screen python3 main_toso.py --total_epochs 30 --learning_rate 0.001 --source mnistm --target mnist --channel_size 1 --model_dir ./model_m2mm/run_1_to
# ADAPTATION
screen python3 main.py --total_epochs 100 --learning_rate 0.001 --source mnist --target mnistm --channel_size 1 --model_dir ./model_m2mm/run_1