#!/usr/bin/env bash


python3 main.py --total_epochs 100 --source mnist --target mnistm --model_dir ./model_m2mm/run_1

python3 main.py --total_epochs 100 --learning_rate 0.001 --step target --source svhn --truncate_mnist False --target mnist --channel_size 1 --source_model ./model_s2m/run_10/source_model --target_model ./model_s2m/run_10/adversarial_model