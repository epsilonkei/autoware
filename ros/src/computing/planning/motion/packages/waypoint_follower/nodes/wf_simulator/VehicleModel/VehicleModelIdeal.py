#!/usr/bin/env python
# -*- coding: utf-8 -*-

from VehicleModelAbstract import VehicleModelAbstract
import numpy as np

class VehicleModelIdealSteer(VehicleModelAbstract):
    __IDX = (__X,
             __Y,
             __YAW) = range(0,3)
    __IDX_U = (__VX_DES,
               __STEER_DES) = range(0,2)

    def __init__(self, wheelbase):
        # TODO need to fix it when move to python 3
        super(VehicleModelIdealSteer, self).__init__(dimx = 3, dimu = 2)
        self.__wheelbase = wheelbase

    def calcModel(self, _state, _input):
        yaw = _state[self.__IDX[self.__YAW]]
        vel = _input[self.__IDX_U[self.__VX_DES]]
        steer = _input[self.__IDX_U[self.__STEER_DES]]
        d_state = np.zeros(self.dimx_)
        d_state[self.__IDX[self.__X]] = vel * np.cos(yaw)
        d_state[self.__IDX[self.__Y]] = vel * np.sin(yaw)
        d_state[self.__IDX[self.__YAW]] = vel * np.tan(steer) / self.__wheelbase
        return d_state

class VehicleModelIdealSteerCustomize(VehicleModelAbstract):
    __IDX = (__X,
             __Y,
             __YAW) = range(0,3)
    __IDX_U = (__VX_DES,
               __STEER_DES) = range(0,2)

    def __init__(self, wheelbase):
        # TODO need to fix it when move to python 3
        super(VehicleModelIdealSteerCustomize, self).__init__(dimx = 3, dimu = 2)
        self.__wheelbase = wheelbase
        self.__vel = 0.0
        self.__steer = 0.0

    def calcModel(self, _state, _input): # result return is not depend on _input
        yaw = _state[self.__IDX[self.__YAW]]
        d_state = np.zeros(self.dimx_)
        d_state[self.__IDX[self.__X]] = self.__vel * np.cos(yaw)
        d_state[self.__IDX[self.__Y]] = self.__vel * np.sin(yaw)
        d_state[self.__IDX[self.__YAW]] = self.__vel * np.tan(self.__steer) / self.__wheelbase
        return d_state

    def setVelocity(self, vel):
        self.__vel = vel

    def setSteer(self, steer):
        self.__steer = steer

    def getVelocity(self):
        return self.__vel

    def getSteer(self, steer):
        return self.__steer
