#!/usr/bin/env python3

import re
import importlib
oyaml_spec = importlib.util.find_spec("oyaml")
if oyaml_spec is not None:
    import oyaml as yaml
else:
    import yaml

def parsePOS(filename):
    lines = open(filename).read().splitlines()
    nbLines = len(lines)
    idxStart = next(i for i,l in enumerate(lines) if l[0] != '%')
    keys = lines[idxStart - 1][1:].split()
    fpPattern = '''-?\d*\.\d*'''
    intPattern = '''-?\d+'''
    key2pattern = {
        'GPST':             '''\d{4}/\d+/\d+\ +\d+:\d+:\d+\.\d+''',
        'latitude(deg)':    fpPattern,
        'longitude(deg)':   fpPattern,
        'height(m)':        fpPattern,
        'Q':                intPattern,
        'ns':               intPattern,
        'sdn(m)':           fpPattern,
        'sde(m)':           fpPattern,
        'sdu(m)':           fpPattern,
        'sdne(m)':          fpPattern,
        'sdeu(m)':          fpPattern,
        'sdun(m)':          fpPattern,
        'age(s)':           fpPattern,
        'ratio':            fpPattern
    }
    linePattern = re.compile('\ +'.join(['(%s)' % key2pattern[key] for key in keys]))
    toNum = lambda k, v: int(v) if key2pattern[k] is intPattern else float(v) if key2pattern[k] is fpPattern else v
    result = []
    for l in lines[idxStart:]:
        matches = re.match(linePattern, l)
        if matches is not None:
            values = list(matches.groups())
            assert len(values) == len(keys)
            result.append(dict([(k,toNum(k, v)) for k, v in zip(keys, values)]))
    return result



