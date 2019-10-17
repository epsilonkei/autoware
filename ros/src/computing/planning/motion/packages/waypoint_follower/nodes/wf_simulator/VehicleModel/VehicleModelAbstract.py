#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class VehicleModelAbstract(object):
    __metaclass__ = ABCMeta

    def __init__ (self, dimx, dimu):
        self.dimx_ = dimx
        self.dimu_ = dimu
        self.state_ = np.zeros(dimx)
        self.input_ = np.zeros(dimu)

    @abstractmethod
    def calcModel(self, _state, _input):
        raise NotImplementedError

    def updateEuler(self, dt, _input):
        self.state_ += self.calcModel(self.state_, _input) * dt

    def updateRungeKutta(self, dt, _input):
        k1 = self.calcModel(self.state_, _input)
        k2 = self.calcModel(self.state_ + k1 * 0.5 * dt, _input)
        k3 = self.calcModel(self.state_ + k2 * 0.5 * dt, _input)
        k4 = self.calcModel(self.state_ + k3 * dt, _input)
        self.state_ += 1.0/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt

    def setState(self, _state):
        self.state_ = _state

    def setInput(self, _input):
        self.input_ = _input

    def getState(self):
        return self.state_

    def getInput(self):
        return self.input_
