import numpy as np

import pathlib
import sys


def make(theta_range, py):
    pathlib.Path('./scripts').mkdir(parents=True, exist_ok=True)
    preamble = open('preamble_synth.txt').read()
    for theta in theta_range:
        for ds_num in [1, 2, 3]:
            with open('./scripts/ds%d_%0.2f.sh' % (ds_num, theta), 'w') as f:
                f.write(preamble\
                    .replace('!ds_num!', '%d' % ds_num)\
                    .replace('!theta!', '%0.2f' % theta)\
                    .replace('!pyargs!', '%s %0.2f %d' % (py, theta, ds_num)))

            
if __name__ == '__main__':
    make(theta_range=[round(e, 2) for e in np.arange(0.01, 1.0, 0.01)], 
         py=sys.argv[1])
