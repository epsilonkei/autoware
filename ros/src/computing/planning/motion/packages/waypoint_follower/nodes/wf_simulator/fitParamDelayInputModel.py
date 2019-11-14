#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import sys
from rosbag2csv import basename_to_csv, rel2abs
try:
    import pandas as pd
except ImportError:
    print ('Please install pandas. See http://pandas.pydata.org/pandas-docs/stable/')
    sys.exit(1)

FREQ_SAMPLE = 0.001

# low pass filter
def lowpass_filter(data, cutoff_freq=2, order=1, dt=FREQ_SAMPLE):
    tau = 1.0 / (2 * np.pi * cutoff_freq)
    for _ in range(order):
        for i in range(1,len(data)):
            data[i] = (tau / (tau + dt) * data[i-1] + dt / (tau + dt) * data[i])
    return data

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
                                delay, args, speed_type = False):
    tm_cmd, cmd_delay = getCmdValueWithDelay(cmd_data, delay)
    tm_act, act, dact = getActValue(act_data, speed_type)
    _t_min = max(tm_cmd[0], tm_act[0])
    tm_cmd = tm_cmd - _t_min
    tm_act = tm_act - _t_min
    if args.upper_cutoff_time > 0:
        _t_max = min(tm_cmd[-1], tm_act[-1], args.upper_cutoff_time)
    else:
        _t_max = min(tm_cmd[-1], tm_act[-1])
    MAX_CNT = int((_t_max - args.lower_cutoff_time) / FREQ_SAMPLE)
    dact_samp = [None] * MAX_CNT
    diff_actcmd_samp = [None] * MAX_CNT
    ind_cmd = 0
    ind_act = 0
    for ind in range(MAX_CNT):
        ti = ind * FREQ_SAMPLE + args.lower_cutoff_time
        while (tm_cmd[ind_cmd + 1] < ti):
            ind_cmd += 1
        cmd_delay_i = getLinearInterpolate(tm_cmd, cmd_delay, ind_cmd, ti)
        while (tm_act[ind_act + 1] < ti):
            ind_act += 1
        act_i = getLinearInterpolate(tm_act, act, ind_act, ti)
        dact_i = getLinearInterpolate(tm_act, dact, ind_act, ti)
        dact_samp[ind] = dact_i
        diff_actcmd_samp[ind] = act_i - cmd_delay_i
    dact_samp = np.array(dact_samp)
    diff_actcmd_samp = np.array(diff_actcmd_samp)
    if args.cutoff_freq > 0:
        dact_samp = lowpass_filter(dact_samp, cutoff_freq=args.cutoff_freq)
        diff_actcmd_samp = lowpass_filter(diff_actcmd_samp, cutoff_freq=args.cutoff_freq)
    dact_samp = dact_samp.reshape(1,-1)
    diff_actcmd_samp = diff_actcmd_samp.reshape(1,-1)
    tau = -np.dot(diff_actcmd_samp, np.linalg.pinv(dact_samp))[0,0]
    error = np.linalg.norm(diff_actcmd_samp + tau * dact_samp) / dact_samp.shape[1]
    return tau, error

def getFittingParam(cmd_data, act_data, args, speed_type = False):
    delay_range = int((args.max_delay - args.min_delay) / args.delay_incr)
    delays = [args.min_delay + i * args.delay_incr for i in range(delay_range + 1)]
    error_min = 1.0e10
    delay_opt = -1
    tau_opt = -1
    for delay in delays:
        tau, error = getFittingTimeConstantParam(cmd_data, act_data, delay, args, speed_type=speed_type)
        if tau > 0:
            if error < error_min:
                error_min = error
                delay_opt = delay
                tau_opt = tau
        else:
            break
    return tau_opt, delay_opt, error_min

def getDataFromLog(basename):
    pd_data = [None] * len(topics)
    for i, topic in enumerate(topics):
        csv_log = basename_to_csv(rel2abs(args.basename), topic)
        pd_data[i] = pd.read_csv(csv_log, sep=' ')
    return pd_data

if __name__ == '__main__':
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed']
    parser = argparse.ArgumentParser(description='Paramter fitting for Input Delay Model (First Order System with Dead Time) with rosbag file input')
    parser.add_argument('--basename', type=str, help='rosbag basename')
    parser.add_argument('--lower_cutoff_time', default=0.0, type=float, help='Lower cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time (default is 0.0)')
    parser.add_argument('--upper_cutoff_time', default=-1.0, type=float, help='Upper cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time, minus value for set upper cutoff_time as the end of bag file (default is -1.0)')
    parser.add_argument('--cutoff_freq', default=0.1, type=float, help='Cutoff freq for low-pass filter[Hz], negative value will disable low-pass filter (default is 0.1)')
    parser.add_argument('--min_delay', default=0.1, type=float, help='Min value for searching delay loop (default is 0.1)')
    parser.add_argument('--max_delay', default=1.0, type=float, help='Max value for searching delay loop (default is 1.0)')
    parser.add_argument('--delay_incr', default=0.01, type=float, help='Step value for searching delay loop (default is 0.01)')
    args = parser.parse_args()

    pd_data = getDataFromLog(args.basename)
    tau_opt, delay_opt, error = getFittingParam(pd_data[0], pd_data[1], args, speed_type=False)
    print ('Steer angle: tau_opt = %2.4f, delay_opt = %2.4f, error = %2.4e' %(tau_opt, delay_opt, error))
    tau_opt, delay_opt, error = getFittingParam(pd_data[2], pd_data[3], args, speed_type=True)
    print ('Velocity   : tau_opt = %2.4f, delay_opt = %2.4f, error = %2.4e' %(tau_opt, delay_opt, error))
