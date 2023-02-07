#!/bin/sh

python main.py --data_path ./test_data.pkl --task_name icews_transe --kge 'TComplEx' --gpu cuda:0 --gamma 0.001 --reg 0.01 --lr 0.001
