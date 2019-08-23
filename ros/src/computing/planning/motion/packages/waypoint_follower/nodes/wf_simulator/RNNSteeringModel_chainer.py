#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_DTYPE'] = 'float64'  # set environment variable for using float64 in chainer

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers
from chainer import serializers
import random
from WFSimulatorCore import WFSimulator
from fitParamDelayInputModel import rosbag_to_csv, rel2abs
import sys
try:
    import pandas as pd
except ImportError:
    print ('Please install pandas. See http://pandas.pydata.org/pandas-docs/stable/')
    sys.exit(1)

# import cupy

f_model = './saved_model'
for ele in [f_model]:
    if not os.path.exists(ele):
        os.mkdir(ele)

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

class RNN(Chain):
    def __init__(self, n_input, n_units, n_output):
        super(RNN, self).__init__(
            l1 = L.Linear(n_input, n_units),
            l2 = L.LSTM(n_units, n_units),
            l3 = L.Linear(n_units, n_output),
        )

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)

class RNNSteeringModel(Chain):
    def __init__(self, predictor, _physModel, cutoff_time = 0.0, onlySim = False):
        super(RNNSteeringModel, self).__init__(predictor=predictor)
        self.physModel = _physModel
        self.__onlySim = onlySim
        self._cutoff_time = cutoff_time
        self.__kVel = 1e-4
        self.__kSteer = 1 - self.__kVel

    def predictNextState(self, state):
        # Calc physics function #TODO: need to run prevSimulate previously
        self.physModel.simulateOneStep()
        physPred = self.physModel.getVehicleState()
        if self.__onlySim:
            nextState = physPred
        else:
            _state = state.reshape(1, -1) # TODO
            RNNpred = self.predictor(_state)
            nextState = RNNpred.reshape(1, -1) + physPred
        return nextState

    def __call__(self, state, _actValue):
        # state = self.physModel.getVehicleState()
        _nextState = self.predictNextState(state)
        nextState = _nextState.reshape(-1,1)
        # _actValue = self.physModel.calcLinearInterpolateActValue()
        actValue = _actValue.reshape(-1,1)
        ''' nextState[3] for VELOCITY, nextState[4] for STEERING_ANGLE '''
        vel_loss = F.mean_squared_error(nextState[3], actValue[0])
        steer_loss = F.mean_squared_error(nextState[4], actValue[1])
        loss = self.__kVel * vel_loss + self.__kSteer * steer_loss
        return loss

if __name__ == '__main__':
    # Read data from csv
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed']
    pd_data = [None] * len(topics)
    # argparse
    parser = argparse.ArgumentParser(description='wf simulator using Deep RNN with rosbag file input')
    parser.add_argument('--bag_file', '-b', required=True, type=str, help='rosbag file', metavar='file')
    parser.add_argument('--cutoff_time', '-c', default=0.0, type=float, help='Cutoff time[sec], Parameter fitting will only consider data from t= cutoff_time to the end of the bag file (default is 1.0)')
    parser.add_argument('--demo', '-d', action='store_true', default=False,
                        help='--demo for load save_weights and test predict model')
    parser.add_argument('--onlySim', '-o', action='store_true', default=False,
                        help='--onlySim for disable using RNN predict')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed [0, 2 ** 32), negative value to not set seed, default value is 0')
    args = parser.parse_args()
    if args.seed >= 0:
        set_random_seed(args.seed)

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
    '''
    RNN parameter: n_input, n_units, n_output
    n_input = n_output = size of state
    In this case, with TimeDelaySteerModel, state = [x, y, yaw, vx, steer]
    -> size of state = 5
    '''
    predictor = RNN(5, 10, 5)
    model = RNNSteeringModel(predictor, wfSim, onlySim = args.onlySim)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    ''' ======================================== '''
    def updateModel(_model, train=True):
        loss = 0.0
        iter_cnt = 0
        _model.physModel.prevSimulate()
        while _model.physModel.isSimulateEpochFinish():
            state = _model.physModel.getVehicleState()
            actValue = _model.physModel.calcLinearInterpolateActValue()
            if model.physModel.isInCutoffTime():
                _ = _model(state, actValue)
            else:
                loss += _model(state, actValue)
                iter_cnt += 1
        if train:
            optimizer.target.zerograds()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        return loss/iter_cnt
    ''' ======================================== '''
    if not args.demo:
        for epoch in range(args.epoch):
            model.predictor.reset_state()
            loss = updateModel(model, train=True)
            print ('Epoch: %2d, loss: %2.6e'%(epoch, loss.data))
        serializers.save_npz(os.path.join(f_model, "RNNSteeringModel_chainer.npz"), model)
    else:
        serializers.load_npz(os.path.join(f_model, "RNNSteeringModel_chainer.npz"), model)
        # Test
        model.predictor.reset_state()
        loss = updateModel(model, train=False)
        print ('Test loss: %2.6e' %(loss.data))
        model.physModel.wrapSimStateAct()
        model.physModel.plotSimulateResult()
