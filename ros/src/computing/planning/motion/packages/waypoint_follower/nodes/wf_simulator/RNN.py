#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
import random
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
            l3 = L.LSTM(n_hidden, n_hidden),
            l4 = L.LSTM(n_hidden, n_hidden),
            l5 = L.Linear(n_hidden, n_output),
        )

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)
