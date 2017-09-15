#!/usr/bin/python

import sys
import re
import math


def get_pitch(fname):
    fname = re.sub(r'/wav/', '/f0/', fname)
    fname = re.sub(r'.wav$', '.pitch', fname)
    pitch = []
    with open(fname) as pfile:
        for line in pfile:
            val = float(line.rstrip().split()[0])
            val = 0 if val == 0 else math.log(val)
            pitch.append(val)
    return pitch


with open('slt.txt') as f:
    for line in f:
        labs = line.rstrip().split()
        fname = labs.pop(0)
        user = labs.pop(0)
        labs.pop(0)
        print fname, user, ':',
        for lab in labs:
            print lab,
        print ':',
        pitch = get_pitch(fname)
        assert len(pitch) < len(labs)
        lastv = pitch[-1]
        for i in xrange(len(pitch), len(labs)):
            pitch.append(lastv)

        for p in pitch:
            print '{:.4f}'.format(float(p)),
        print
