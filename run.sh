#!/bin/bash

srun -p short -N 1 -n 1 -c 8 --pty --export=ALL --mem=10Gb --time=08:00:00 /bin/bash
cd /home/chakravarty.s/csye7374-chakravarty.s/homework3/parallel_ml
python3 part1_process.py
exit
