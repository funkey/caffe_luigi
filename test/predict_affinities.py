from __future__ import print_function
import sys
import os

def predict_affinities(a, b, c, d, gpu):
    print("predict affinities stdout")
    print("predict affinities stderr", file=sys.stderr)
    os.system('echo predict affinities echo stdout')
    os.system('>&2 echo predict affinities echo stderr')
