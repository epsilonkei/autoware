#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import panda as pd

def loadCSV(path):
    # Read data
    df = pd.read_csv(path)
    # Time convert
    time_index = pd.DatetimeIndex((df["%time"]))
    df.set_index(time_index, inplace=True)
    # remove useless data
    df.drop(['%time', 'field.header.seq', 'field.header.stamp'], axis=1, inplace=True)
    # remove offset
    for state in df.keys():
        df[state] = df[state] - df[state].values[0]
    # add data
    df["d"] = np.sqrt(df.x ** 2 + df.y ** 2)  # distance
    return df

def getFittingParam(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paramter fitting for Input Delay Model (First Order System with Dead Time)')
    parser.add_argument('--cmd_log', '-c', required=True, help='Vehicle command log file')
    parser.add_argument('--act_log', '-a', required=True, help='Vehicle actual status log fil')
    args = parser.parse_args()
    getFittingParam(args)
