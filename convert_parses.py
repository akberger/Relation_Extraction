import sys
import os
import glob
from subprocess import call

"""
Usage:
    python convert_parses.py <PS parse dir> <dependency parse dir>
"""

ps_dir, dep_dir = sys.argv[1], sys.argv[2]
ps_parses = glob.glob(os.path.join(ps_dir,'*.parse'))

if not os.path.exists(dep_dir):
    os.makedirs(dep_dir)

for parse in ps_parses:
    depparse = os.path.join(dep_dir, os.path.basename(parse))
    call('java -jar pennconverter.jar -f {} -t {}.dep'.format(parse, depparse),
         shell=True)

