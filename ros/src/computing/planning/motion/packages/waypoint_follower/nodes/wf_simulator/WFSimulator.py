#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rosbag2csv import basename_to_csv, rel2abs
from WFSimulatorCore import WFSimulator, getYawFromQuaternion
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
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed', \
               'current_pose/pose']
    pd_data = [None] * len(topics)
    parser = argparse.ArgumentParser(description='wf simulator python Implement with rosbag file input')
    parser.add_argument('--basename', '-b', required=True, type=str, help='rosbag file basename', metavar='file')
    parser.add_argument('--lower_cutoff_time', default=0.0, type=float, help='Lower cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time (default is 0.0)')
    parser.add_argument('--upper_cutoff_time', default=-1.0, type=float, help='Upper cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time, minus value for set upper cutoff_time as the end of bag file (default is -1.0)')
    args = parser.parse_args()
    for i, topic in enumerate(topics):
        csv_log = basename_to_csv(rel2abs(args.basename), topic)
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
    ##
    px0 = pd_data[4]['field.position.x'][0]
    py0 = pd_data[4]['field.position.y'][0]
    pz0 = pd_data[4]['field.position.z'][0]
    ox0 = pd_data[4]['field.orientation.x'][0]
    oy0 = pd_data[4]['field.orientation.y'][0]
    oz0 = pd_data[4]['field.orientation.z'][0]
    ow0 = pd_data[4]['field.orientation.w'][0]
    yaw0 = getYawFromQuaternion((ox0, oy0, oz0, ow0))
    v0 = vel_act[0]
    steer0 = steer_act[0]
    # Create WF simulator instance + intialize (if necessary)
    wfSim = WFSimulator(loop_rate = 50.0, wheel_base = 2.7)
    wfSim.parseData(tm_cmd, input_cmd, tm_act, state_act, args.lower_cutoff_time, args.upper_cutoff_time)
    wfSim.prevSimulate((px0, py0, yaw0, v0, steer0))
    # Run simulate
    wfSim.simulate()
    # simulate results MeanSquaredError
    mse_vel, mse_steer = wfSim.MeanSquaredError()
    print ('Velocity Mean Squared Error:       %2.6e'%(mse_vel))
    print ('Steering Angle Mean Squared Error: %2.6e'%(mse_steer))
    print ('Velocity + Steering Angle Mean Squared Error: %2.6e'%(0.5*(mse_vel + mse_steer)))
    # Plot simulate results
    wfSim.plotSimulateResult()
