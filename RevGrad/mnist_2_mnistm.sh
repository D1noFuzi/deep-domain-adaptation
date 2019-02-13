#!/usr/bin/env bash


python3 main.py --total_epochs 100 --source mnist --target mnistm --model_dir ./model_m2mm/run_1
python3 main.py --learning_rate 0.001 --total_epochs 10 --source_only True --source mnist --target mnistm --model_dir ./model_m2mm/run_9_so
python3 main.py --learning_rate 0.001 --total_epochs 10 --source_only True --truncate_mnist False --source svhn --target mnistm --model_dir ./model_s2m/run_10_so