#!/usr/bin/env python
# -*- coding: utf-8 -*-

from VehicleModel.VehicleModelTimeDelay import VehicleModelTimeDelaySteer
import numpy as np
import matplotlib.pyplot as plt
import bisect
import tf

def getLinearInterpolate(tm0, tm1, val0, val1, tm, __EPS=1e-9):
    if (tm1 - tm0) < __EPS:
        return val0
    else:
        val = val0 + (val1 - val0) / (tm1 - tm0) * (tm - tm0)
        return val

def getYawFromQuaternion(quaternion):
    return tf.transformations.euler_from_quaternion(quaternion)[2]

class WFSimulator(object):
    __VehicleModelType = (__IDEAL_TWIST,
                          __IDEAL_STEER,
                          __DELAY_TWIST,
                          __DELAY_STEER,
                          __CONST_ACCEL_TWIST,
                          __IDEAL_FORKLIFT_RLS,
                          __DELAY_FORKLIFT_RLS) = range(0,7)

    def __init__(self,
                 loop_rate, wheel_base,
                 vehicle_model = 'DELAY_STEER'):
        self.__wheel_base = wheel_base
        self.__dt = 1.0 / loop_rate
        self.__lower_cutoff_time = 0.0
        self.__upper_cutoff_time = np.inf
        self.tm_cmd = None
        self.input_cmd = None
        self.tm_act = None
        self.state_act = None
        self.__STEER_THRES = 0.05 # Desgined Parameter
        self.visual_pts = []
        # WFSimulator parameter, #TODO: it should be able to set by using rosparam
        vel_lim = 40.0
        steer_lim = 3.14 / 3.0
        vel_delay = 0.25
        vel_time_const = 0.6197
        steer_delay = 0.1
        steer_time_const = 0.1142
        # Simulation result list
        self.sim_state_act = []
        if vehicle_model == 'DELAY_STEER':
            self.__vehicle_model_type = self.__VehicleModelType[self.__DELAY_STEER]
            self.__vehicle_model = VehicleModelTimeDelaySteer(vel_lim, steer_lim,
                                                              self.__wheel_base, self.__dt,
                                                              vel_delay, vel_time_const,
                                                              steer_delay, steer_time_const)
        else:
            raise NotImplementedError
        # For update simulation result list
        self.__prev_tm = 0.0
        # self.__prev_state = self.__vehicle_model.getState() # this will involve bug because of deep copy
        self.__prev_state = self.__vehicle_model.getState().copy()
        # For simulate
        self.__tm = 0.0
        self.__tm_end = 0.0
        self.__ind_cmd = 0
        self.__ind_act = 0

    def parseData(self, _tm_cmd, _input_cmd, _tm_act, _state_act,
                  _lower_cutoff_time, _upper_cutoff_time):
        self.tm_cmd = _tm_cmd
        self.input_cmd = _input_cmd
        self.tm_act = _tm_act
        self.state_act = _state_act
        self.__lower_cutoff_time = _lower_cutoff_time
        self.__upper_cutoff_time = _upper_cutoff_time if _upper_cutoff_time > 0.0 else np.inf

    def calcLinearInterpolateActValue(self):
        if self.__ind_act >= len(self.tm_act):
            act_state = getLinearInterpolate(self.tm_act[-2],
                                             self.tm_act[-1],
                                             self.state_act[:,-2],
                                             self.state_act[:,-1],
                                             self.__tm + self.__dt)
        else:
            act_state = getLinearInterpolate(self.tm_act[self.__ind_act - 1],
                                             self.tm_act[self.__ind_act],
                                             self.state_act[:,self.__ind_act - 1],
                                             self.state_act[:,self.__ind_act],
                                             self.__tm)
        return act_state

    def calcLinearInterpolateNextActValue(self):
        if self.__ind_act >= len(self.tm_act):
            act_state = getLinearInterpolate(self.tm_act[-2],
                                             self.tm_act[-1],
                                             self.state_act[:,-2],
                                             self.state_act[:,-1],
                                             self.__tm + self.__dt)
        elif self.__tm + self.__dt < self.tm_act[self.__ind_act] or \
             self.__ind_act + 1 >= len(self.tm_act):
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

    def setInitialState(self, _state):
        self.__vehicle_model.setState(np.array(_state))

    def updateSimulationActValue(self, state):
        if self.__ind_act < len(self.tm_act):
            nextActTm = self.tm_act[self.__ind_act]
            if self.__tm >= nextActTm:
                act_state = getLinearInterpolate(self.__prev_tm, self.__tm,
                                                 self.__prev_state, state,
                                                 nextActTm)
                self.sim_state_act.append(act_state)
                self.__ind_act += 1

    def prevSimulate(self, init_state):
        self.setInitialState(init_state)
        # Clear simulation result list
        self.sim_state_act = []
        self.visual_pts = []
        self.__tm = min(self.tm_cmd[0], self.tm_act[0])
        # Remove offset in tm_cmd and tm_act
        self.tm_cmd -= self.__tm
        self.tm_act -= self.__tm
        self.__tm = 0.0
        # ##
        self.__tm_end = max(self.tm_cmd[-1], self.tm_act[-1])
        self.__prev_tm = self.__tm
        self.__prev_state = self.__vehicle_model.getState().copy()
        self.__ind_cmd = 0
        self.__ind_act = 0
        # Reset Vehicle Model
        if self.__vehicle_model_type == self.__VehicleModelType[self.__DELAY_STEER]:
            self.__vehicle_model.initializeInputQueue(self.__dt)
        self.updateVehicleCmd()
        self.updateSimulationActValue(self.__vehicle_model.getState())

    def getVehicleState(self):
        return self.__vehicle_model.getState()

    def getVehicleInputCmd(self):
        return self.__vehicle_model.getInput()

    def getDeltaT(self):
        return self.__dt

    def savePrevState(self):
        self.__prev_state = self.__vehicle_model.getState().copy()

    def updateVehicleCmd(self):
        if self.__ind_cmd < len(self.tm_cmd) and self.__tm >= self.tm_cmd[self.__ind_cmd]:
            self.__vehicle_model.setInput(self.input_cmd[:, self.__ind_cmd])
            self.__ind_cmd += 1

    def updateVehicleModelState(self, _state):
        self.__vehicle_model.setState(_state)

    def calcVehicleState(self):
        self.__vehicle_model.update(self.__dt)

    def updateSimulationTime(self):
        self.__prev_tm = self.__tm
        self.__tm += self.__dt

    def simulateOneStep(self):
        self.savePrevState()
        self.calcVehicleState()
        self.updateSimulationTime()
        self.updateSimulationActValue(self.__vehicle_model.getState())
        self.updateVehicleCmd()

    def isSimulateEpochFinish(self):
        return self.__tm > self.__tm_end

    def isInCutoffTime(self):
        return self.__tm < self.__lower_cutoff_time or self.__tm > self.__upper_cutoff_time

    def isLowFriction(self):
        _input = self.getVehicleInputCmd()
        return abs(_input[1]) > self.__STEER_THRES #__STEER_DES := 2

    def addVisualPoint(self):
        self.visual_pts.append(self.__tm)

    def wrapSimStateAct(self):
        if self.__ind_act < len(self.tm_act):
            self.updateSimulationActValue(self.__vehicle_model.getState())
        self.sim_state_act = np.array(self.sim_state_act)

    def simulate(self):
        while not self.isSimulateEpochFinish():
            self.simulateOneStep()
        self.wrapSimStateAct()

    def MeanSquaredError(self):
        assert len(self.sim_state_act) != 0, 'Simulate should be run before calculating MeanSquaredError'
        cutoff_index = bisect.bisect_left(self.tm_act, self.__cutoff_time)
        mse_vel = (np.square(self.sim_state_act[cutoff_index:,3] - self.state_act[0,cutoff_index:])).mean(axis=0)
        mse_steer = (np.square(self.sim_state_act[cutoff_index:,4] - self.state_act[1,cutoff_index:])).mean(axis=0)
        return mse_vel, mse_steer

    def plotSimulateResult(self):
        if self.__vehicle_model_type == self.__VehicleModelType[self.__DELAY_STEER]:
            fig = plt.figure(figsize=(16, 16))
            ax1 = fig.add_subplot(211)
            ax1.plot(self.tm_cmd, self.input_cmd[0], label='vehicle_cmd/velocity')
            ax1.plot(self.tm_act, self.state_act[0], label='vehicle_status/velocity')
            ax1.plot(self.tm_act, self.sim_state_act[:,3], # __VX
                     label='sim_vehicle_status/velocity')
            ax1.set_ylabel("Velocity [m/s]")
            ax1.set_xlabel("Time [s]")
            ax1.legend(loc='best')
            ax2 = fig.add_subplot(212, sharex=ax1)
            ax2.plot(self.tm_cmd, self.input_cmd[1], label='vehicle_cmd/steer')
            ax2.plot(self.tm_act, self.state_act[1], label='vehicle_status/steer')
            ax2.plot(self.tm_act, self.sim_state_act[:,4], # __STEER
                     label='sim_vehicle_status/steer')
            ax2.set_ylabel("Steering Angle [rad]")
            ax2.set_xlabel("Time [s]")
            ax2.legend(loc='best')
            fig.tight_layout()
            plt.show()
            fig.savefig('simulate_result.png')
        else:
            raise NotImplementedError

    def plotSimulateResultIncludeDsteer(self):
        if self.__vehicle_model_type == self.__VehicleModelType[self.__DELAY_STEER]:
            fig = plt.figure(figsize=(16, 16))
            ax1 = fig.add_subplot(311)
            ax1.plot(self.tm_cmd, self.input_cmd[0], label='vehicle_cmd/velocity')
            ax1.plot(self.tm_act, self.state_act[0], label='vehicle_status/velocity')
            ax1.plot(self.tm_act, self.sim_state_act[:,3], # __VX
                     label='sim_vehicle_status/velocity')
            ax1.set_ylabel("Velocity [m/s]")
            ax1.set_xlabel("Time [s]")
            ax1.legend(loc='best')
            ax2 = fig.add_subplot(312, sharex=ax1)
            ax2.plot(self.tm_cmd, self.input_cmd[1], label='vehicle_cmd/steer')
            ax2.plot(self.tm_act, self.state_act[1], label='vehicle_status/steer')
            ax2.plot(self.tm_act, self.sim_state_act[:,4], # __STEER
                     label='sim_vehicle_status/steer')
            if len(self.visual_pts) > 0:
                self.visual_pts = np.array(self.visual_pts)
                zeros = np.zeros(len(self.visual_pts))
                ax2.scatter(self.visual_pts, zeros, s=5)
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
            ax3.plot(self.tm_act[1:-1], differential(self.sim_state_act[:,4], self.tm_act),
                     label='sim_vehicle_status/dsteer') # d(__STEER)
            ax3.set_ylabel("Steering Angle Velocity [rad/sec]")
            ax3.set_xlabel("Time [s]")
            ax3.legend(loc='best')
            fig.tight_layout()
            plt.show()
            fig.savefig('simulate_result.png')
        else:
            raise NotImplementedError
