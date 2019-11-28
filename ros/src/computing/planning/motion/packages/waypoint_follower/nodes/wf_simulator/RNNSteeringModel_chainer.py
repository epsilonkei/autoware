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

    def predictNextState(self, RNNinput, onlySim = False):
        '''
        return RNN predict + Physics function
        @override of WFSimulator.simulateOneStep()
        '''
        # Save prev_state
        self.physModel.savePrevState()
        # Calculate Vehicle State (Eg: Runge-Kutta)
        self.physModel.calcVehicleState()
        physPred = self.physModel.getVehicleState()
        if onlySim:
            nextState = physPred
        else:
            _RNNinput = RNNinput.reshape(1, -1) # TODO
            RNNpred = self.predictor(_RNNinput)
            nextState = F.concat((np.array([[0.0, 0.0, 0.0, 0.0]]), RNNpred)) + physPred
        # Update simulation time
        self.physModel.updateSimulationTime()
        # Update Simulation Act
        if onlySim:
            self.physModel.updateSimulationActValue(nextState)
        else:
            self.physModel.updateSimulationActValue(nextState.data.flatten())
        if not onlySim:
            # Update Vehicle Model State for next calcVehicleModel iteration
            self.physModel.updateVehicleModelState(nextState.data.flatten())
        # Update Vehicle Cmd
        self.physModel.updateVehicleCmd()
        return nextState

    def __call__(self, state, _nextActValue, onlySim = False):
        # state = self.physModel.getVehicleState()
        _nextState = self.predictNextState(state, onlySim = onlySim)
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

def evaluateModel(basename, lower_cutoff_time, upper_cutoff_time):
    tm_cmd, input_cmd, tm_act, state_act, init_state = getDataFromLog(basename)
    model.physModel.parseData(tm_cmd, input_cmd, tm_act, state_act,
                              lower_cutoff_time, upper_cutoff_time)
    model.predictor.reset_state()
    vel_loss, steer_loss, dsteer_loss = updateModel(model, init_state, train=False)
    return vel_loss, steer_loss, dsteer_loss

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='wf simulator using Deep RNN with rosbag file input')
    parser.add_argument('--basename', type=str, help='Log basename')
    parser.add_argument('--datcfg', type=str, help='Training data config', metavar='file')
    parser.add_argument('--lower_cutoff_time', default=0.0, type=float, help='Lower cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time (default is 0.0)')
    parser.add_argument('--upper_cutoff_time', default=-1.0, type=float, help='Upper cutoff time[sec], Parameter fitting will only consider data from t=lower_cutoff_time to t=upper_cutoff_time, minus value for set upper cutoff_time as the end of bag file (default is -1.0)')
    parser.add_argument('--demo', '-d', action='store_true', default=False,
                        help='--demo for test predict model')
    parser.add_argument('--load', type=str, default='', help='--load for load saved_model')
    parser.add_argument('--onlySim', '-o', action='store_true', default=False,
                        help='--onlySim for disable using RNN predict')
    parser.add_argument('--onlyHighFric', action='store_true', default=False,
                        help='--onlySim for apply RNN predict for only High Friction period')
    parser.add_argument('--noGTInput', action='store_true', default=False,
                        help='--noGTInput for training RNN with input from previous simulated output')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--save_eps', type=int, default=10,
                        help='Save model every args.save_eps epochs')
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
            loss = batch_steer_loss # + 0.01 * batch_dsteer_loss
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        while not _model.physModel.isSimulateEpochFinish():
            inputCmd = model.physModel.getVehicleInputCmd()
            # RNN input = [v, steer, v_d, steer_d]
            if not train or args.noGTInput:
                actState = _model.physModel.calcLinearInterpolateActValue()
                RNNinput = np.concatenate([actState, inputCmd])
            else:
                state = _model.physModel.getVehicleState()
                RNNinput = np.concatenate([state[3:5], inputCmd])
            nextActValue = _model.physModel.calcLinearInterpolateNextActValue()
            if model.physModel.isInCutoffTime() or \
               args.onlyHighFric and model.physModel.isLowFriction():
                if args.onlyHighFric:
                    _ , _, _ = _model(RNNinput, nextActValue, onlySim=True)
                else:
                    _ , _, _ = _model(RNNinput, nextActValue, onlySim=args.onlySim)
            else:
                iter_vel_loss, iter_steer_loss, iter_dsteer_loss = _model(RNNinput, nextActValue, onlySim=args.onlySim)
                model.physModel.addVisualPoint()
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
        if iter_cnt > 0:
            return all_vel_loss/iter_cnt, all_steer_loss/iter_cnt, all_dsteer_loss/iter_cnt
        else:
            return chainer.Variable(np.array([0.0])), chainer.Variable(np.array([0.0])), chainer.Variable(np.array([0.0]))
    ''' ======================================== '''
    if not args.demo:
        # Training mode
        if args.datcfg:
            with open(args.datcfg, 'r') as f:
                cfg = yaml.load(f)
                data_list = cfg['train']
                test = cfg['test']
            tm_cmds = []
            input_cmds = []
            tm_acts = []
            state_acts = []
            init_states = []
            for data in data_list:
                _tm_cmd, _input_cmd, _tm_act, _state_act, _init_state = getDataFromLog(data['basename'])
                tm_cmds.append(_tm_cmd)
                input_cmds.append(_input_cmd)
                state_acts.append(_state_act)
                tm_acts.append(_tm_act)
                init_states.append(_init_state)
        else:
            tm_cmd, input_cmd, tm_act, state_act, init_state = getDataFromLog(args.basename)
            model.physModel.parseData(tm_cmd, input_cmd, tm_act, state_act,
                                      args.lower_cutoff_time, args.upper_cutoff_time)
        log_folder = time.strftime('%Y%m%d%H%M%S') + '_' + args.log_suffix
        f_result = log_folder
        f_model = log_folder + '/saved_model'
        for ele in [f_result, f_model]:
            if not os.path.exists(ele):
                os.makedirs(ele)
        saveCodeStatus(f_result)

        train_log = open(os.path.join(f_result, 'train_log.txt'), mode='w')
        test_log = open(os.path.join(f_result, 'test_log.txt'), mode='w')
        for epoch in range(1, args.epoch + 1):
            model.predictor.reset_state()
            if args.datcfg:
                ind = random.randrange(len(data_list))
                model.physModel.parseData(tm_cmds[ind], input_cmds[ind], tm_acts[ind], state_acts[ind], \
                                          data_list[ind]['lower_cutoff_time'],
                                          data_list[ind]['upper_cutoff_time'])
                vel_loss, steer_loss, dsteer_loss = updateModel(model, init_states[ind], train=True)
            else:
                vel_loss, steer_loss, dsteer_loss = updateModel(model, init_state, train=True)
            print ('Epoch: %4d, Velocity loss: %2.6e, Steer loss: %2.6e, dSteer loss: %2.6e'%(epoch, vel_loss.data, steer_loss.data, dsteer_loss.data))
            train_log.write('%4d %2.6e %2.6e %2.6e\n'%(epoch, vel_loss.data, steer_loss.data,
                                                       dsteer_loss.data))
            if epoch % args.save_eps == 0:
                # Test with test data and save model
                if args.datcfg:
                    vel_loss, steer_loss, dsteer_loss = evaluateModel(test[0]['basename'],
                                                                      test[0]['lower_cutoff_time'], test[0]['upper_cutoff_time'])
                else:
                    vel_loss, steer_loss, dsteer_loss = evaluateModel(args.basename, args.lower_cutoff_time, args.upper_cutoff_time)
                print ('Epoch: %4d, Test velocity loss: %2.6e, Test steer loss: %2.6e, Test dsteer loss: %2.6e'%(epoch, vel_loss.data, steer_loss.data, dsteer_loss.data))
                test_log.write('%4d %2.6e %2.6e %2.6e\n'%(epoch, vel_loss.data, steer_loss.data,
                                                          dsteer_loss.data))
                serializers.save_npz(os.path.join(f_model, '%3d.npz'%(epoch)), model)
        train_log.close()
        test_log.close()
    else:
        # Test mode
        vel_loss, steer_loss, dsteer_loss = evaluateModel(args.basename, args.lower_cutoff_time, args.upper_cutoff_time)
        print ('Test velocity loss: %2.6e, Test steer loss: %2.6e, Test dsteer loss: %2.6e'%(vel_loss.data, steer_loss.data, dsteer_loss.data))
        model.physModel.wrapSimStateAct()
        model.physModel.plotSimulateResultIncludeDsteer(_visual_pts = args.onlyHighFric)
