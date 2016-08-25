from __future__ import print_function
import sys
import os

def create_segmentations(a, b, c, d, e):
    print("create segmentations stdout")
    print("create segmentations stderr", file=sys.stderr)
    os.system('echo create segmentations echo stdout')
    os.system('>&2 echo create segmentations echo stderr')
