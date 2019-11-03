#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess
import os

def saveCodeStatus(log_folder):
    with open(os.path.join(log_folder, 'cmd_memo'), mode='w') as _memo:
        _memo.write('--------------- Execute command: ---------------\n')
        _memo.write('python %s\n'%(' '.join(sys.argv)))
    subprocess.call(['./shell/save_code_status.sh', log_folder])
