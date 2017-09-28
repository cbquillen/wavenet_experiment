#!/usr/bin/python
import sys
from wavenet import compute_overlap

if len(sys.argv) != 2:
    print >> sys.stderr, "Usage: context {paramfile}"
    sys.exit(1)

opts = type('opts', (), {})
with open(sys.argv[1]) as f:
    exec(f)

print "Context is", compute_overlap(opts), "samples."
