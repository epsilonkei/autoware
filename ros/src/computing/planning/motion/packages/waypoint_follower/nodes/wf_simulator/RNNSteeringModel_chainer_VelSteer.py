#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_DTYPE'] = 'float64'  # set environment variable for using float64 in chainer

import argparse
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers
from chainer import serializers
import random
from fitParamDelayInputModel import rosbag_to_csv, rel2abs
from VehicleModel.VehicleModelIdeal import VehicleModelIdealSteerCustomize
from WFSimulatorCore import getLinearInterpolate
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

    def __call__(self, x): # TODO: add activation function as relu or tanh
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)

class RNNSteeringModel(Chain):
    def __init__(self, predictor,
                 _tm_cmd, _input_cmd,
                 _tm_act, _state_act,
                 loop_rate, wheel_base, cutoff_time = 0.0):
        super(RNNSteeringModel, self).__init__(predictor=predictor)
        self.__wheel_base = wheel_base
        self.__dt = 1.0 / loop_rate
        self.__cutoff_time = cutoff_time
        self.tm_cmd = _tm_cmd
        self.input_cmd = _input_cmd
        self.tm_act = _tm_act
        self.state_act = _state_act
        self.__input_u = np.zeros(2) # input_u = [v_des, steer_des]
        self.__output = np.zeros(4) # output = [v, dv, steer, dsteer]
        self.__input_x = np.zeros(6) # input_x = concat(output, input_u)
        self.__vehicle_model = VehicleModelIdealSteerCustomize(self.__wheel_base)
        # Simulation result list
        self.sim_state_act = []
        # For simulate
        self.__tm = 0.0
        self.__tm_end = 0.0
        self.__ind_cmd = 0
        self.__ind_act = 0
        # For dsteer
        self.__prev_steer = None
        self.__prev_act_steer = None

    ## ==== Feature of WFSimulator ==== ##
    def calcLinearInterpolateActValue(self):
        act_state = getLinearInterpolate(self.tm_act[self.__ind_act - 1],
                                         self.tm_act[self.__ind_act],
                                         self.state_act[:,self.__ind_act - 1],
                                         self.state_act[:,self.__ind_act],
                                         self.__tm)
        return act_state

    def calcLinearInterpolateNextActValue(self):
        if self.__tm + self.__dt < self.tm_act[self.__ind_act] or self.__ind_act < len(self.tm_act):
            act_state = getLinearInterpolate(self.tm_act[self.__ind_act - 1],
                                             self.tm_act[self.__ind_act],
                                             self.state_act[:,self.__ind_act - 1],
                                             self.state_act[:,self.__ind_act],
                                             self.__tm + self.__dt)
        else:
            act_state = getLinearInterpolate(self.tm_act[self.__ind_act],
                                             self.tm_act[self.__ind_act + 1],
                                             self.state_act[:,self.__ind_act],
                                             self.state_act[:,self.__ind_act + 1],
                                             self.__tm + self.__dt)
        return act_state

    def setInitialState(self): # TODO: using geometry_msg Pose and Twist
        x = 0.0
        y = 0.0
        yaw = 0.0
        state = np.array((x, y, yaw))
        self.__vehicle_model.setState(state)

    def updateSimulationActValue(self):
        self.sim_state_act.append(self.__output) # sim_state_act = [v, dv, steer, dsteer]

    def prevSimulate(self):
        self.setInitialState()
        # Clear simulation result list
        self.sim_state_act = []
        self.__tm = min(self.tm_cmd[0], self.tm_act[0])
        # Remove offset in tm_cmd and tm_act
        self.tm_cmd -= self.__tm
        self.tm_act -= self.__tm
        self.__tm = 0.0
        # ##
        self.__tm_end = max(self.tm_cmd[-1], self.tm_act[-1])
        self.__ind_cmd = 0
        self.__ind_act = 0
        self.updateSimulationActValue()

    def updateCmdValue(self):
        if self.__ind_cmd < len(self.tm_cmd) and self.__tm >= self.tm_cmd[self.__ind_cmd]:
            self.__input_u = self.input_cmd[:, self.__ind_cmd]
            self.__ind_cmd += 1

    def updateSimulationTime(self):
        self.__tm += self.__dt

    def isSimulateEpochFinish(self):
        return self.__tm < self.__tm_end

    def isInCutoffTime(self):
        return self.__tm < self.__cutoff_time

    def wrapSimStateAct(self):
        self.sim_state_act = np.array(self.sim_state_act)

    def plotSimulateResultIncludeDsteer(self):
        fig = plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(311)
        ax1.plot(self.tm_cmd, self.input_cmd[0], label='vehicle_cmd/velocity')
        ax1.plot(self.tm_act, self.state_act[0], label='vehicle_status/velocity')
        ax1.plot(self.tm_act, self.sim_state_act[:,0], # Velocity
                 label='sim_vehicle_status/velocity')
        ax1.set_ylabel("Velocity [m/s]")
        ax1.set_xlabel("Time [s]")
        ax1.legend(loc='best')
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax2.plot(self.tm_cmd, self.input_cmd[1], label='vehicle_cmd/steer')
        ax2.plot(self.tm_act, self.state_act[1], label='vehicle_status/steer')
        ax2.plot(self.tm_act, self.sim_state_act[:,2], # Steer
                 label='sim_vehicle_status/steer')
        ax2.set_ylabel("Steering Angle [rad]")
        ax2.set_xlabel("Time [s]")
        ax2.legend(loc='best')
        def differential(state, tm):
            return (state[2:] - state[:-2]) / (tm[2:] - tm[:-2])
        # leap-frog scheme: df(N) = (f(N+1) - f(N-1)) / (t(N+1) - t(N-1))
        '''
        Notes: rosbag time stamp may be have duplicated time,
        so leap-frog scheme is better than explicit scheme and implicit scheme
        '''
        ax3 = fig.add_subplot(313, sharex=ax1)
        ax3.plot(self.tm_act[1:-1], differential(self.state_act[1], self.tm_act),
                 label='vehicle_status/dsteer')
        ax3.plot(self.tm_act[1:-1], differential(self.sim_state_act[:,2], self.tm_act),
                 label='sim_vehicle_status/dsteer') # d(Steer)
        ax3.set_ylabel("Steering Angle Velocity [rad/sec]")
        ax3.set_xlabel("Time [s]")
        ax3.legend(loc='best')
        fig.tight_layout()
        plt.show()
        fig.savefig('RNNVelSteer_simulate_result.png')
    ## ====================================== ##

    def getInputX(self):
        return self.__input_x

    def getOutput(self):
        return self.__output

    def fphys(self, input_x): # input_x = [v, dv, steer, dsteer]
        next_v = input_x[0] + input_x[1] * self.__dt # v[i+1] = v[i] + dv[i] * dt
        next_steer = input_x[2] + input_x[3] * self.__dt # steer[i+1] = steerx[i] + dsteer[i] * dt
        return np.array([next_v, 0.0, next_steer, 0.0])

    def simulateOneStep(self, input_x): # return output
        # Update Cmd Value
        self.updateCmdValue()
        # Calc physic function
        fphys = self.fphys(input_x)
        # Calc rnn output
        _input_x = input_x.reshape(1, -1) # TODO
        frnn = self.predictor(_input_x)
        output = frnn.reshape(1, -1) + fphys
        # Update simulation time
        self.updateSimulationTime()
        ##
        # # Update for x,y,yaw (if in SIMULATOR mode)
        # self.__vehicle_model.setVelocity(nextState[0]) # vel
        # self.__vehicle_model.setState(nextState[2]) # steer
        # # Calculate Vehicle State (Eg: Runge-Kutta)
        # self.__vehicle_model.updateRungeKutta(self.__dt)
        ##
        # Update input_x for next simulateOneStep iteration
        self.__output = output.data.flatten()
        self.__input_x = np.concatenate([self.__output, self.__input_u])
        return output

    def __call__(self, input_x, _actValue):
        _output = self.simulateOneStep(input_x)
        output = _output.reshape(-1,1)
        actValue = _actValue.reshape(-1,1)
        '''
        output[0] for VELOCITY, output[2] for STEERING_ANGLE, output[3] for DSTEERING
        Notes: this DSTEERING is not differential of STEERING_ANGLE
        '''
        vel_loss = F.mean_squared_error(output[0], actValue[0])
        steer_loss = F.mean_squared_error(output[2], actValue[1])
        if self.__prev_steer is not None and self.__prev_act_steer is not None:
            dsteer_loss = F.mean_squared_error((output[2] - self.__prev_steer) / self.__dt,
                                               (actValue[1] - self.__prev_act_steer) / self.__dt)
        else:
            dsteer_loss = chainer.Variable(np.array([0.0]))
        self.__prev_steer = output[2]
        self.__prev_act_steer = actValue[1]
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
    parser.add_argument('--demo', '-d', action='store_true', default=False,
                        help='--demo for test predict model')
    parser.add_argument('--load', type=str, default='', help='--load for load saved_model')
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
    '''
    RNN parameter: n_input, n_units, n_output
    n_input = size of input_x + size_of input_u, n_output = size of input_x,
    In this case, input_x = [v, dv, steer, dsteer], input_u = [v_des, steer_des]
    -> n_input = 6, n_output = 4
    '''
    predictor = RNN(6, 10, 4)
    model = RNNSteeringModel(predictor, tm_cmd, input_cmd, tm_act, state_act,
                             loop_rate = 50.0, wheel_base = 2.7, cutoff_time = args.cutoff_time)
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
        _model.prevSimulate()
        def __runOptimizer():
            optimizer.target.zerograds()
            loss = batch_vel_loss + batch_steer_loss + batch_dsteer_loss
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        while _model.isSimulateEpochFinish():
            input_x = _model.getInputX()
            actValue = _model.calcLinearInterpolateNextActValue()
            if model.isInCutoffTime():
                _ , _, _ = _model(input_x, actValue)
            else:
                iter_vel_loss, iter_steer_loss, iter_dsteer_loss = _model(input_x, actValue)
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
        model.wrapSimStateAct()
        model.plotSimulateResultIncludeDsteer()
