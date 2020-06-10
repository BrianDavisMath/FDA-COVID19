#!/usr/bin/env bash

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.0 -d False -t False -n 'all_features'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.1 -d False -t False -n 'drop_top_10_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.2 -d False -t False -n 'drop_top_20_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.3 -d False -t False -n 'drop_top_30_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.4 -d False -t False -n 'drop_top_40_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.5 -d False -t False -n 'drop_top_50_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.6 -d False -t False -n 'drop_top_60_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.7 -d False -t False -n 'drop_top_70_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.8 -d False -t False -n 'drop_top_80_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 0.9 -d False -t False -n 'drop_top_90_percent'

python job.py -a 0.25 -s 0.05 -m -0.05 -z 1.0 -d False -t False -n 'drop_top_100_percent'