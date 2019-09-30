#!/bin/bash

nohup python3 main.py --train true --train_file ./data/train3.txt --dev_file ./data/dev3.txt > train.log &

