#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_DTYPE'] = 'float64'  # set environment variable for using float64 in chainer

import argparse
import numpy as np
import chainer
import chainer.functions as F
from chainer import Chain, optimizers, serializers
from WFSimulatorCore import WFSimulator, getYawFromQuaternion
from rosbag2csv import basename_to_csv, rel2abs
from RNN import RNN, set_random_seed
import time
import sys
import yaml
from saveCodeStatus import saveCodeStatus
import random

try:
    import pandas as pd
except ImportError:
    print ('Please install pandas. See http://pandas.pydata.org/pandas-docs/stable/')
    sys.exit(1)

# import cupy

class RNNSteeringModel(Chain):
    def __init__(self, predictor, _physModel, onlySim = False):
        super(RNNSteeringModel, self).__init__(predictor=predictor)
        self.physModel = _physModel
        self.__onlySim = onlySim
        self.__prev_steer = None
        self.__prev_act_steer = None

    def resetPrevSteer(self):
        self.__prev_steer = None
        self.__prev_act_steer = None

    def predictNextState(self, RNNinput):
        '''
        return RNN predict + Physics function
        @override of WFSimulator.simulateOneStep()
        '''
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
            nextState = F.concat((np.array([[0.0, 0.0, 0.0, 0.0]]), RNNpred)) + physPred
        # Update simulation time
        self.physModel.updateSimulationTime()
        # Update Simulation Act
        if self.__onlySim:
            self.physModel.updateSimulationActValue(nextState)
        else:
            self.physModel.updateSimulationActValue(nextState.data.flatten())
        if not self.__onlySim:
            # Update Vehicle Model State for next calcVehicleModel iteration
            self.physModel.updateVehicleModelState(nextState.data.flatten())
        # Update Vehicle Cmd
        self.physModel.updateVehicleCmd()
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

def getDataFromLog(basename):
    if basename[0] == '~':
        basename = os.path.expanduser(basename)
    else:
        basename = rel2abs(basename)
    for i, topic in enumerate(topics):
        csv_log = basename_to_csv(basename, topic)
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
    return tm_cmd, input_cmd, tm_act, state_act, (px0, py0, yaw0, v0, steer0)

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='wf simulator using Deep RNN with rosbag file input')
    parser.add_argument('--basename', type=str, help='Log basename')
    parser.add_argument('--datcfg', type=str, help='Training data config', metavar='file')
    parser.add_argument('--cutoff_time', '-c', default=0.0, type=float, help='Cutoff time[sec], Parameter fitting will only consider data from t= cutoff_time to the end of the bag file (default is 1.0)')
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

    # Read data from csv and data.cfg
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed', \
               'current_pose/pose']
    pd_data = [None] * len(topics)

    # Create WF simulator instance + intialize (if necessary)
    wfSim = WFSimulator(loop_rate = 50.0, wheel_base = 2.7)
    '''
    RNN parameter: n_input, n_units, n_output
    Input = [v, steer, v_d, steer_d], output = [steer]
    -> n_input = 4, n_output = 1
    '''
    predictor = RNN(4, 32, 1)
    model = RNNSteeringModel(predictor, wfSim, onlySim = args.onlySim)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    if args.load:
        serializers.load_npz(args.load, model)

    ''' ======================================== '''
    def updateModel(_model, init_state, train=True):
        all_vel_loss = 0.0
        all_steer_loss = 0.0
        all_dsteer_loss = 0.0
        iter_cnt = 0
        batch_vel_loss = 0.0
        batch_steer_loss = 0.0
        batch_dsteer_loss = 0.0
        batch_cnt = 0
        ## Reset state
        _model.physModel.prevSimulate(init_state)
        _model.resetPrevSteer()
        def __runOptimizer():
            optimizer.target.zerograds()
            # loss = batch_vel_loss + batch_steer_loss + batch_dsteer_loss
            loss = batch_steer_loss + 0.01 * batch_dsteer_loss
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        while _model.physModel.isSimulateEpochFinish():
            state = _model.physModel.getVehicleState()
            inputCmd = model.physModel.getVehicleInputCmd()
            # RNN input = [v, steer, v_d, steer_d]
            RNNinput = np.concatenate([state[3:5], inputCmd])
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
                batch_dsteer_loss = 0.0
                batch_cnt = 0
        # Update model using remain part data
        if train and batch_cnt > 0:
            __runOptimizer()
        return all_vel_loss/iter_cnt, all_steer_loss/iter_cnt, all_dsteer_loss/iter_cnt
    ''' ======================================== '''
    if not args.demo:
        # Training mode
        if args.datcfg:
            with open(args.datcfg, 'r') as f:
                data_list = yaml.load(f)
            tm_cmds = input_cmds = tm_acts = state_acts = init_states = [None] * len(data_list)
            for i, data in enumerate(data_list['logs']):
                tm_cmds[i], input_cmds[i], tm_acts[i], state_acts[i], init_states[i] = getDataFromLog(data['basename'])
        else:
            tm_cmd, input_cmd, tm_act, state_act, init_state = getDataFromLog(args.basename)
            model.physModel.parseData(tm_cmd, input_cmd, tm_act, state_act, args.cutoff_time)
        log_folder = time.strftime("%Y%m%d%H%M%S") + '_' + args.log_suffix
        f_result = log_folder
        f_model = log_folder + '/saved_model'
        for ele in [f_result, f_model]:
            if not os.path.exists(ele):
                os.makedirs(ele)
        saveCodeStatus(f_result)

        with open(os.path.join(f_result, 'train_log.txt'), mode='w') as log:
            for epoch in range(args.epoch):
                model.predictor.reset_state()
                if args.datcfg:
                    ind = random.randrange(len(data_list))
                    model.physModel.parseData(tm_cmds[ind], input_cmds[ind], tm_acts[ind], state_acts[ind], \
                                              data_list['logs'][ind]['cutoff_time'])
                    vel_loss, steer_loss, dsteer_loss = updateModel(model, init_states[ind], train=True)
                else:
                    vel_loss, steer_loss, dsteer_loss = updateModel(model, init_state, train=True)
                print ('Epoch: %4d, Velocity loss: %2.6e, Steer loss: %2.6e, dSteer loss: %2.6e'%(epoch, vel_loss.data, steer_loss.data, dsteer_loss.data))
                log.write('%4d %2.6e %2.6e %2.6e\n'%(epoch, vel_loss.data, steer_loss.data,
                                               dsteer_loss.data))
        serializers.save_npz(os.path.join(f_model, "RNNSteeringModel_chainer.npz"), model)
    else:
        # Test mode
        tm_cmd, input_cmd, tm_act, state_act, init_state = getDataFromLog(args.basename)
        model.physModel.parseData(tm_cmd, input_cmd, tm_act, state_act, args.cutoff_time)
        model.predictor.reset_state()
        vel_loss, steer_loss, dsteer_loss = updateModel(model, init_state, train=False)
        print ('Test velocity loss: %2.6e, Test steer loss: %2.6e, Test dsteer loss: %2.6e'%(vel_loss.data, steer_loss.data, dsteer_loss.data))
        model.physModel.wrapSimStateAct()
        model.physModel.plotSimulateResultIncludeDsteer()
