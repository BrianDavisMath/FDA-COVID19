#!/usr/bin/env bash

python job.py -a 0.02 -s 0.01 -f test_data/ -d False -t False -n 'no_weights'

python job.py -a 0.02 -s 0.01 -f test_data/ -d True -t False -n 'dim_red_weights'

python job.py -a 0.02 -s 0.01 -f test_data/ -d False -t True -n 'training_weights'

python job.py -a 0.02 -s 0.01 -f test_data/ -d True -t True -n 'all_weights'