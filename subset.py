#!/usr/bin/python

# Subset VCTK with the users in ulist

ulist = {}
with open('ulist') as f:
    for line in f:
        line = line.rstrip()
        ulist[int(line)] = len(ulist)

with open('align3_tph_f0.txt') as f:
    for line in f:
        fields = line.rstrip().split()
        user = int(fields[1])
        if user in ulist:
            print fields[0], ulist[user], " ".join(fields[2:])
