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
import time
import sys
try:
    import pandas as pd
except ImportError:
    print ('Please install pandas. See http://pandas.pydata.org/pandas-docs/stable/')
    sys.exit(1)

# import cupy

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

class RNN(Chain):
    def __init__(self, n_input, n_hidden, n_output):
        super(RNN, self).__init__(
            l1 = L.Linear(n_input, n_hidden),
            l2 = L.LSTM(n_hidden, n_hidden),
            l3 = L.Linear(n_hidden, n_output),
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
        self.__prev_steer = None
        self.__prev_act_steer = None

    def predictNextState(self, RNNinput):
        '''
        return RNN predict + Physics function
        @override of WFSimulator.simulateOneStep()
        '''
        # Update Vehicle Cmd
        self.physModel.updateVehicleCmd()
        # Save prev_state
        self.physModel.savePrevState()
        # Calculate Vehicle State (Eg: Runge-Kutta)
        self.physModel.calcVehicleState()
        physPred = self.physModel.getVehicleState()
        if self.__onlySim:
            nextState = physPred
        else:
            _RNNinput = RNNinput.reshape(1, -1) # TODO
            RNNpred = self.predictor(_RNNinput)
            nextState = RNNpred.reshape(1, -1) + physPred
        # Update simulation time
        self.physModel.updateSimulationTime()
        # Update Simulation Act
        if self.__onlySim:
            self.physModel.updateSimulationActValue(nextState)
        else:
            self.physModel.updateSimulationActValue(nextState.data.flatten())
            # Update Vehicle Model State for next calcVehicleModel iteration
            self.physModel.updateVehicleModelState(nextState.data.flatten())
        return nextState

    def __call__(self, state, _nextActValue):
        # state = self.physModel.getVehicleState()
        _nextState = self.predictNextState(state)
        nextState = _nextState.reshape(-1,1)
        # _actValue = self.physModel.calcLinearInterpolateActValue()
        nextActValue = _nextActValue.reshape(-1,1)
        ''' nextState[3] for VELOCITY, nextState[4] for STEERING_ANGLE '''
        vel_loss = F.mean_squared_error(nextState[3], nextActValue[0])
        steer_loss = F.mean_squared_error(nextState[4], nextActValue[1])
        if self.__prev_steer is not None and self.__prev_act_steer is not None:
            dsteer_loss = F.mean_squared_error((nextState[4] - self.__prev_steer) / self.physModel.getDeltaT(), (nextActValue[1] - self.__prev_act_steer) / self.physModel.getDeltaT())
        else:
            dsteer_loss = chainer.Variable(np.array([0.0]))
        self.__prev_steer = nextState[4]
        self.__prev_act_steer = nextActValue[1]
        return vel_loss, steer_loss, dsteer_loss

if __name__ == '__main__':
    # Read data from csv
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed']
    pd_data = [None] * len(topics)
    # argparse
    parser = argparse.ArgumentParser(description='wf simulator using Deep RNN with rosbag file input')
    parser.add_argument('--bag_file', required=True, type=str, help='rosbag file', metavar='file')
    parser.add_argument('--cutoff_time', '-c', default=0.0, type=float, help='Cutoff time[sec], Parameter fitting will only consider data from t= cutoff_time to the end of the bag file (default is 1.0)')
    parser.add_argument('--RNNarch', type=str, default='InputOnlyVelSteer',
                        choices=('InputOnlyState', 'IncludeXYyaw', 'InputOnlyVelSteer'))
    parser.add_argument('--demo', '-d', action='store_true', default=False,
                        help='--demo for test predict model')
    parser.add_argument('--load', type=str, default='', help='--load for load saved_model')
    parser.add_argument('--onlySim', '-o', action='store_true', default=False,
                        help='--onlySim for disable using RNN predict')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch for update training model')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed [0, 2 ** 32), negative value to not set seed, default value is 0')
    parser.add_argument('--log_suffix', '-l', default='', type=str, help='Saving log folder suffix')
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
    if args.RNNarch == 'InputOnlyVelSteer':
        '''
        RNN parameter: n_input, n_units, n_output
        n_input = 2 (vx, steer) + size_of input, n_output = size of state,
        In this case, with TimeDelaySteerModel, state = [x, y, yaw, vx, steer], input = [v_d, steer_d]
        -> n_input = 4, n_output = 5
        '''
        predictor = RNN(4, 10, 5)
    elif args.RNNarch == 'IncludeXYyaw':
        '''
        RNN parameter: n_input, n_units, n_output
        n_input = size of state + size_of input, n_output = size of state,
        In this case, with TimeDelaySteerModel, state = [x, y, yaw, vx, steer], input = [v_d, steer_d]
        -> n_input = 7, n_output = 5
        '''
        predictor = RNN(7, 10, 5)
    elif args.RNNarch == 'InputOnlyState':
        '''
        RNN parameter: n_input, n_units, n_output
        n_input = n_output = size of state,
        In this case, with TimeDelaySteerModel, state = [x, y, yaw, vx, steer], input = [v_d, steer_d]
        -> n_input = 5, n_output = 5
        '''
        predictor = RNN(5, 10, 5)
    model = RNNSteeringModel(predictor, wfSim, onlySim = args.onlySim)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    if args.load:
        serializers.load_npz(args.load, model)

    ''' ======================================== '''
    def updateModel(_model, train=True):
        all_vel_loss = 0.0
        all_steer_loss = 0.0
        all_dsteer_loss = 0.0
        iter_cnt = 0
        batch_vel_loss = 0.0
        batch_steer_loss = 0.0
        batch_dsteer_loss = 0.0
        batch_cnt = 0
        _model.physModel.prevSimulate()
        def __runOptimizer():
            optimizer.target.zerograds()
            loss = batch_vel_loss + batch_steer_loss + batch_dsteer_loss
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        while _model.physModel.isSimulateEpochFinish():
            state = _model.physModel.getVehicleState()
            inputCmd = model.physModel.getVehicleInputCmd()
            if args.RNNarch == 'InputOnlyVelSteer':
                # RNN input = [v, steer, v_d, steer_d]
                RNNinput = np.concatenate([state[3:5], inputCmd])
            elif args.RNNarch == 'IncludeXYyaw':
                # RNN input = [x, y, yaw, v, steer, v_d, steer_d]
                RNNinput = np.concatenate([state, inputCmd])
            elif args.RNNarch == 'InputOnlyState':
                # RNN input = [x, y, yaw, v, steer]
                RNNinput = state
            nextActValue = _model.physModel.calcLinearInterpolateNextActValue()
            if model.physModel.isInCutoffTime():
                _ , _, _ = _model(RNNinput, nextActValue)
            else:
                iter_vel_loss, iter_steer_loss, iter_dsteer_loss = _model(RNNinput, nextActValue)
                all_vel_loss += iter_vel_loss
                all_steer_loss += iter_steer_loss
                all_dsteer_loss += iter_dsteer_loss
                batch_vel_loss += iter_vel_loss
                batch_steer_loss += iter_steer_loss
                batch_dsteer_loss += iter_dsteer_loss
                iter_cnt += 1
                batch_cnt += 1
            if train and batch_cnt == args.batch:
                __runOptimizer()
                # reset batch loss
                batch_vel_loss = 0.0
                batch_steer_loss = 0.0
                batch_cnt = 0
        # Update model using remain part data
        if train and batch_cnt > 0:
            __runOptimizer()
        return all_vel_loss/iter_cnt, all_steer_loss/iter_cnt, all_dsteer_loss/iter_cnt
    ''' ======================================== '''
    if not args.demo:
        log_folder = time.strftime("%Y%m%d%H%M%S") + '_' + args.log_suffix
        f_result = log_folder
        f_model = log_folder + '/saved_model'
        for ele in [f_result, f_model]:
            if not os.path.exists(ele):
                os.makedirs(ele)

        with open(os.path.join(f_result, 'train_log.txt'), mode='w') as log:
            for epoch in range(args.epoch):
                model.predictor.reset_state()
                vel_loss, steer_loss, dsteer_loss = updateModel(model, train=True)
                print ('Epoch: %4d, Velocity loss: %2.6e, Steer loss: %2.6e, dSteer loss: %2.6e'%(epoch, vel_loss.data, steer_loss.data, dsteer_loss.data))
                log.write('%4d %2.6e %2.6e %2.6e\n'%(epoch, vel_loss.data, steer_loss.data,
                                               dsteer_loss.data))
        serializers.save_npz(os.path.join(f_model, "RNNSteeringModel_chainer.npz"), model)
    else:
        # Test
        model.predictor.reset_state()
        vel_loss, steer_loss, dsteer_loss = updateModel(model, train=False)
        print ('Test velocity loss: %2.6e, Test steer loss: %2.6e, Test dsteer loss: %2.6e'%(vel_loss.data, steer_loss.data, dsteer_loss.data))
        model.physModel.wrapSimStateAct()
        model.physModel.plotSimulateResultIncludeDsteer()
