#!/usr/bin/env python
# -*- coding: utf-8 -*-

from VehicleModel.VehicleModelTimeDelay import VehicleModelTimeDelaySteer
import numpy as np
import matplotlib.pyplot as plt

def getLinearInterpolate(tm0, tm1, val0, val1, tm, __EPS=1e-9):
    if (tm1 - tm0) < __EPS:
        return val0
    else:
        val = val0 + (val1 - val0) / (tm1 - tm0) * (tm - tm0)
        return val

class WFSimulator(object):
    __VehicleModelType = (__IDEAL_TWIST,
                          __IDEAL_STEER,
                          __DELAY_TWIST,
                          __DELAY_STEER,
                          __CONST_ACCEL_TWIST,
                          __IDEAL_FORKLIFT_RLS,
                          __DELAY_FORKLIFT_RLS) = range(0,7)

    def __init__(self,
                 _tm_cmd, _input_cmd,
                 _tm_act, _state_act,
                 loop_rate, wheel_base,
                 vehicle_model = 'DELAY_STEER'):
        self.__wheel_base = wheel_base
        self.__dt = 1.0 / loop_rate
        self.tm_cmd = _tm_cmd
        self.input_cmd = _input_cmd
        self.tm_act = _tm_act
        self.state_act = _state_act
        # WFSimulator parameter, #TODO: it should be able to set by using rosparam
        vel_lim = 10.0
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
        self.__prev_tm = 0
        self.__prev_state = self.__vehicle_model.getState()

    def updateVehicleCmd(self, input):
        self.__vehicle_model.setInput(input)

    def updateSimulateAct(self, prev_tm, tm, tm_act):
        state = self.__vehicle_model.getState()
        act_state = getLinearInterpolate(prev_tm, tm,
                                         self.__prev_state, state, tm_act)
        return act_state

    def setInitialState(self): # TODO: using geometry_msg Pose and Twist
        x = 0.0
        y = 0.0
        yaw = 0.0
        vx = 0.0
        wz = 0.0
        steer = 0.0
        if self.__vehicle_model_type == self.__VehicleModelType[self.__DELAY_STEER]:
            state = np.array((x, y, yaw, vx, steer))
            self.__vehicle_model.setState(state)
        else:
            raise NotImplementedError

    def simulate(self):
        self.setInitialState()
        # Clear simulation result list
        self.sim_state_act = []
        tm = min(self.tm_cmd[0], self.tm_act[0])
        # Remove offset in tm_cmd and tm_act
        self.tm_cmd -= tm
        self.tm_act -= tm
        tm = 0.0
        # ##
        tm_end = max(self.tm_cmd[-1], self.tm_act[-1])
        self.__prev_tm = tm
        self.__prev_state = self.__vehicle_model.getState()
        ind_cmd = 0
        ind_act = 0
        ''' Update Simulation Act Value Start'''
        nextUpdateActTm = self.tm_act[ind_act]
        if tm >= nextUpdateActTm:
            act_state = self.updateSimulateAct(self.__prev_tm, tm, nextUpdateActTm)
            self.sim_state_act.append(act_state)
            ind_act += 1
        ''' Update Simulation Act Value End'''
        while (tm < tm_end):
            while tm < (self.tm_cmd[ind_cmd] if ind_cmd < len(self.tm_cmd) else tm_end):
                self.__prev_state = self.__vehicle_model.getState()
                self.__vehicle_model.updateRungeKutta(self.__dt)
                tm += self.__dt
                ''' Update Simulation Act Value Start'''
                nextUpdateActTm = self.tm_act[ind_act]
                if tm >= nextUpdateActTm:
                    act_state = self.updateSimulateAct(self.__prev_tm, tm,
                                                       nextUpdateActTm)
                    self.sim_state_act.append(act_state)
                    ind_act += 1
                ''' Update Simulation Act Value End'''
            if (ind_cmd < len(self.tm_cmd)):
                self.updateVehicleCmd(self.input_cmd[:, ind_cmd])
            ind_cmd += 1
        self.sim_state_act = np.array(self.sim_state_act)

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
