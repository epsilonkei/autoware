#!/usr/bin/env python
# -*- coding: utf-8 -*-

from VehicleModelAbstract import VehicleModelAbstract
from collections import deque
import numpy as np

class VehicleModelTimeDelaySteer(VehicleModelAbstract):
    __IDX = (__X,
             __Y,
             __YAW,
             __VX,
             __STEER) = range(0,5)
    __IDX_U = (__VX_DES,
               __STEER_DES) = range(0,2)

    def __init__(self, vel_lim, steer_lim, wheelbase, dt,
                 vel_delay, vel_time_const, steer_delay, steer_time_const):
        # TODO need to fix it when move to python 3
        super(VehicleModelTimeDelaySteer, self).__init__(dimx = 5, dimu = 2)
        self.__vel_lim = vel_lim
        self.__steer_lim = steer_lim
        self.__wheelbase = wheelbase
        self.__vel_delay = vel_delay
        self.__vel_time_const = vel_time_const
        self.__steer_delay = steer_delay
        self.__steer_time_const = steer_time_const
        self.__initializeInputQueue(dt)

    def __initializeInputQueue(self, dt):
        self.__vel_input_queue = deque()
        self.__steer_input_queue = deque()
        vel_input_queue_size = int(round(self.__vel_delay / dt))
        for i in range(vel_input_queue_size):
            self.__vel_input_queue.append(0.0)
        steer_input_queue_size = int(round(self.__steer_delay / dt))
        for i in range(steer_input_queue_size):
            self.__steer_input_queue.append(0.0)

    def calcModel(self, _state, _input):
        vel = _state[self.__IDX[self.__VX]]
        yaw = _state[self.__IDX[self.__YAW]]
        steer = _state[self.__IDX[self.__STEER]]
        delay_input_vel = _input[self.__IDX_U[self.__VX_DES]]
        delay_input_steer = _input[self.__IDX_U[self.__STEER_DES]]
        delay_vx_des = max(min(delay_input_vel, self.__vel_lim), -self.__vel_lim)
        delay_steer_des = max(min(delay_input_steer, self.__steer_lim), -self.__steer_lim)
        vx_rate = - (vel - delay_vx_des) / self.__vel_time_const
        steer_rate = - (steer - delay_steer_des) / self.__steer_time_const
        d_state = np.zeros(self.dimx_)
        d_state[self.__IDX[self.__X]] = vel * np.cos(yaw)
        d_state[self.__IDX[self.__Y]] = vel * np.sin(yaw)
        d_state[self.__IDX[self.__YAW]] = vel * np.tan(steer) / self.__wheelbase
        d_state[self.__IDX[self.__VX]] = vx_rate
        d_state[self.__IDX[self.__STEER]] = steer_rate
        return d_state

    def update(self, dt):
        delay_input = np.zeros(self.dimu_)
        self.__vel_input_queue.append(self.input_[self.__IDX_U[self.__VX_DES]])
        delay_input_vel = self.__vel_input_queue.popleft()
        self.__steer_input_queue.append(self.input_[self.__IDX_U[self.__STEER_DES]])
        delay_input_steer = self.__steer_input_queue.popleft()
        delay_input[self.__IDX_U[self.__VX_DES]] = delay_input_vel
        delay_input[self.__IDX_U[self.__STEER_DES]] = delay_input_steer
        self.updateRungeKutta(dt, delay_input)
