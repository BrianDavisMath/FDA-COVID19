#!/usr/bin/env bash

python job.py -a 0.85 -s 0.1 -m 0.05 -d False -t False -n 'no_weights'

python job.py -a 0.85 -s 0.1 -m 0.05 -d False -t True -n 'training_weights'

