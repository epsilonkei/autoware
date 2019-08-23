#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fitParamDelayInputModel import rosbag_to_csv, rel2abs
from WFSimulatorCore import WFSimulator
import argparse
import numpy as np
import sys
try:
    import pandas as pd
except ImportError:
    print ('Please install pandas. See http://pandas.pydata.org/pandas-docs/stable/')
    sys.exit(1)

if __name__ == '__main__':
    # Read data from csv
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed']
    pd_data = [None] * len(topics)
    parser = argparse.ArgumentParser(description='wf simulator python Implement with rosbag file input')
    parser.add_argument('--bag_file', '-b', required=True, type=str, help='rosbag file', metavar='file')
    parser.add_argument('--cutoff_time', '-c', default=0.0, type=float, help='Cutoff time[sec], Parameter fitting will only consider data from t= cutoff_time to the end of the bag file (default is 1.0)')
    args = parser.parse_args()
    for i, topic in enumerate(topics):
        csv_log = rosbag_to_csv(rel2abs(args.bag_file), topic)
        pd_data[i] = pd.read_csv(csv_log, sep=' ')
    tm_cmd = np.array(list(pd_data[0]['%time'])) * 1e-9
    steer_cmd = np.array(list(pd_data[0]['field']))
    vel_cmd = np.array(list(pd_data[2]['field']))
    input_cmd = np.vstack((vel_cmd, steer_cmd))
    ##
    tm_act = np.array(list(pd_data[1]['%time'])) * 1e-9
    steer_act = np.array(list(pd_data[1]['field']))
    vel_act = np.array(list(pd_data[3]['field'])) / 3.6 # km/h -> m/s
    state_act = np.vstack((vel_act, steer_act))
    # Create WF simulator instance + intialize (if necessary)
    wfSim = WFSimulator(tm_cmd, input_cmd, tm_act, state_act,
                        loop_rate = 50.0, wheel_base = 2.7, cutoff_time = args.cutoff_time)
    # wfSim.setInitialState()
    # Run simulate
    wfSim.simulate()
    # simulate results MeanSquaredError
    mse_vel, mse_steer = wfSim.MeanSquaredError()
    print ('Velocity Mean Squared Error:       %2.6e'%(mse_vel))
    print ('Steering Angle Mean Squared Error: %2.6e'%(mse_steer))
    print ('Velocity + Steering Angle Mean Squared Error: %2.6e'%(0.5*(mse_vel + mse_steer)))
    # Plot simulate results
    wfSim.plotSimulateResult()
