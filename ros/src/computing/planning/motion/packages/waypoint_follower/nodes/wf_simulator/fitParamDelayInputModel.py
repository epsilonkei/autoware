#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd

FREQ_SAMPLE = 0.001

def getActValue(path, speed_type):
    # Read data
    df = pd.read_csv(path, sep=' ')
    tm = np.array(list(df['%time'])) * 1e-9
    # Unit Conversion
    if speed_type:
        val = np.array(list(df['field'])) / 3.6
    else:
        val = np.array(list(df['field']))
    # Calc differential
    dval = (val[2:] - val[:-2]) / (tm[2:] - tm[:-2])
    return tm[1:-1], val[1:-1], dval

def getCmdValueWithDelay(path, delay):
    # Read data
    df = pd.read_csv(path, sep=' ')
    tm = np.array(list(df['%time'])) * 1e-9
    val = np.array(list(df['field']))
    return tm + delay, val

def getLinearInterpolate(_tm, _val, _index, ti):
    tmp_t = _tm[_index]
    tmp_nextt = _tm[_index + 1]
    tmp_val = _val[_index]
    tmp_nextval = _val[_index + 1]
    val_i = tmp_val + (tmp_nextval - tmp_val) / (tmp_nextt - tmp_t) * (ti - tmp_t)
    return val_i

def getFittingTimeConstantParam(cmd_log, act_log, delay, speed_type = False):
    tm_cmd, cmd_delay = getCmdValueWithDelay(cmd_log, delay)
    tm_act, act, dact = getActValue(act_log, speed_type)
    _t_min = max(tm_cmd[0], tm_act[0])
    _t_max = min(tm_cmd[-1], tm_act[-1])
    tm_cmd = tm_cmd - _t_min
    tm_act = tm_act - _t_min
    MAX_CNT = int((_t_max - _t_min) / FREQ_SAMPLE)
    dact_samp = [None] * MAX_CNT
    diff_actcmd_samp = [None] * MAX_CNT
    ind_cmd = 0
    ind_act = 0
    for ind in range(MAX_CNT):
        ti = ind * FREQ_SAMPLE
        while (tm_cmd[ind_cmd + 1] < ti):
            ind_cmd += 1
        cmd_delay_i = getLinearInterpolate(tm_cmd, cmd_delay, ind_cmd, ti)
        while (tm_act[ind_act + 1] < ti):
            ind_act += 1
        act_i = getLinearInterpolate(tm_act, act, ind_act, ti)
        dact_i = getLinearInterpolate(tm_act, dact, ind_act, ti)
        dact_samp[ind] = dact_i
        diff_actcmd_samp[ind] = act_i - cmd_delay_i
    dact_samp = np.array(dact_samp).reshape(1,-1)
    diff_actcmd_samp = np.array(diff_actcmd_samp).reshape(1,-1)
    tau = -np.dot(diff_actcmd_samp, np.linalg.pinv(dact_samp))[0,0]
    error = np.linalg.norm(diff_actcmd_samp + tau * dact_samp) / dact_samp.shape[1]
    return tau, error

MIN_DELAY = 0.0
MAX_DELAY = 2.0
DELAY_INCR = 0.1

def getFittingParam(cmd_log, act_log, speed_type = False):
    delay_range = int((MAX_DELAY - MIN_DELAY) / DELAY_INCR)
    delays = [MIN_DELAY + i * DELAY_INCR for i in range(delay_range + 1)]
    error_min = 1.0e10
    delay_opt = -1
    tau_opt = -1
    for delay in delays:
        tau, error = getFittingTimeConstantParam(cmd_log, act_log, delay, speed_type=speed_type)
        if error < error_min:
            error_min = error
            delay_opt = delay
            tau_opt = tau
    return tau_opt, delay_opt, error_min

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paramter fitting for Input Delay Model (First Order System with Dead Time)')
    parser.add_argument('--cmd_log', '-c', required=True, help='Vehicle command log file')
    parser.add_argument('--act_log', '-a', required=True, help='Vehicle actual status log file')
    parser.add_argument('--speed_type', '-s', choices=['True', 'False'], \
                        default='False', help='Type speed or not: Steering Angle = False, velocity = True')
    args = parser.parse_args()
    if args.speed_type == 'True':
        tau_opt, delay_opt, error = getFittingParam(args.cmd_log, args.act_log, speed_type=True)
    else:
        tau_opt, delay_opt, error = getFittingParam(args.cmd_log, args.act_log, speed_type=False)
    print tau_opt, delay_opt, error
