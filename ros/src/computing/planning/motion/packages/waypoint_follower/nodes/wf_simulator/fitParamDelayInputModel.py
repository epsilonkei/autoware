#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd
import subprocess
from os import getcwd
from os.path import dirname, basename, splitext, join, exists

FREQ_SAMPLE = 0.001
CUTOFF_TIME = 15.0

def rel2abs(path):
    '''
    Return absolute path from relative path input
    '''
    return join(getcwd(), path)

def rosbag_to_csv(path, topic_name):
    name = splitext(basename(path))[0]
    suffix = topic_name.replace('/', '-')
    output_path = join(dirname(path), name + "_" + suffix + ".csv")
    if exists(output_path):
        return output_path
    else:
        command = "rostopic echo -b {0} -p /{1} | sed -e 's/,/ /g' > {2}".format(path, topic_name, output_path)
        print command
        subprocess.check_call(command, shell=True)
        return output_path

def getActValue(df, speed_type):
    tm = np.array(list(df['%time'])) * 1e-9
    # Unit Conversion
    if speed_type:
        val = np.array(list(df['field'])) / 3.6
    else:
        val = np.array(list(df['field']))
    # Calc differential
    dval = (val[2:] - val[:-2]) / (tm[2:] - tm[:-2])
    return tm[1:-1], val[1:-1], dval

def getCmdValueWithDelay(df, delay):
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

def getFittingTimeConstantParam(cmd_data, act_data, \
                                delay, speed_type = False):
    tm_cmd, cmd_delay = getCmdValueWithDelay(cmd_data, delay)
    tm_act, act, dact = getActValue(act_data, speed_type)
    _t_min = max(tm_cmd[0], tm_act[0])
    _t_max = min(tm_cmd[-1], tm_act[-1])
    tm_cmd = tm_cmd - _t_min
    tm_act = tm_act - _t_min
    MAX_CNT = int((_t_max - _t_min - CUTOFF_TIME) / FREQ_SAMPLE)
    dact_samp = [None] * MAX_CNT
    diff_actcmd_samp = [None] * MAX_CNT
    ind_cmd = 0
    ind_act = 0
    for ind in range(MAX_CNT):
        ti = ind * FREQ_SAMPLE + CUTOFF_TIME
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

def getFittingParam(cmd_data, act_data, speed_type = False):
    delay_range = int((MAX_DELAY - MIN_DELAY) / DELAY_INCR)
    delays = [MIN_DELAY + i * DELAY_INCR for i in range(delay_range + 1)]
    error_min = 1.0e10
    delay_opt = -1
    tau_opt = -1
    for delay in delays:
        tau, error = getFittingTimeConstantParam(cmd_data, act_data, delay, speed_type=speed_type)
        if error < error_min:
            error_min = error
            delay_opt = delay
            tau_opt = tau
    return tau_opt, delay_opt, error_min

if __name__ == '__main__':
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
             'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed']
    pd_data = [None] * len(topics)
    parser = argparse.ArgumentParser(description='Paramter fitting for Input Delay Model (First Order System with Dead Time) with rosbag file input')
    parser.add_argument('--bag_file', '-b', required=True, type=str, help='rosbag file', metavar='file')
    args = parser.parse_args()

    for i, topic in enumerate(topics):
        csv_log = rosbag_to_csv(rel2abs(args.bag_file), topic)
        pd_data[i] = pd.read_csv(csv_log, sep=' ')
    tau_opt, delay_opt, error = getFittingParam(pd_data[0], pd_data[1], speed_type=False)
    print ('Steer angle: tau_opt = %2.4f, delay_opt = %2.4f, error = %2.4e' %(tau_opt, delay_opt, error))
    tau_opt, delay_opt, error = getFittingParam(pd_data[2], pd_data[3], speed_type=True)
    print ('Velocity   : tau_opt = %2.4f, delay_opt = %2.4f, error = %2.4e' %(tau_opt, delay_opt, error))
