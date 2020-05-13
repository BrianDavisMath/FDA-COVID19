#!/usr/bin/env bash

python job.py -a 0.01 -s 0.01 -d False -t False -n 'no_weights'

python job.py -a 0.01 -s 0.01 -d True -t False -n 'dim_red_weights'

python job.py -a 0.01 -s 0.01 -d False -t True -n 'training_weights'

python job.py -a 0.01 -s 0.01 -d True -t True -n 'all_weights'