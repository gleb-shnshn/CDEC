# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"

import os
import sys

import numpy as np

from algorithms.audio_processing import create_mel_filterbank
from algorithms.ssaec_fast import SSAECFast
from loaders.aec_loader import AECLoader
from utils.mat_helpers import load_numpy_from_mat

sys.path.append(os.path.abspath('../'))


class FeatureGenerator:

    def __init__(self, cache_path):

        self.aec_loader = AECLoader(name='aec')
        self.ssaec = SSAECFast(wlen=512, tail_length=0.250)

        self.fs = self.aec_loader.fs
        self.samples = int(self.fs * 15)
        self.silence = int(self.fs * 5)

        self.dataset_dir = self.aec_loader.dataset_dir
        self.cache_path = cache_path
        self.scenarios = ['nearend', 'farend', 'doubletalk']
        self.modes = ['real', 'simu', 'hard']
        self.train_set_length = 5000
        self.test_set_length = len(self.aec_loader.test)
        self.blind_test_set_length = len(self.aec_loader.test_blind)

        self.nband = 25
        self.Q_long = create_mel_filterbank(nbin=513, fs=16e3, nband=self.nband)
        self.Q_short = create_mel_filterbank(nbin=129, fs=16e3, nband=self.nband)

    # scenarios = ['nearend', 'farend', 'doubletalk']
    # modes = ['real','simu','hard']
    def load_train(self, nbatch=1, mode=None, scenario=None, idx=None,
                   p_modes=(0.3, 0.3, 0.4), p_scenarios=(0.1, 0.1, 0.8)):

        mode0 = mode
        scenario0 = scenario
        idx0 = idx

        x = np.zeros((nbatch, self.samples - self.silence), dtype=np.float32)
        y = np.zeros((nbatch, self.samples - self.silence), dtype=np.float32)
        d = np.zeros((nbatch, self.samples - self.silence), dtype=np.float32)
        e = np.zeros((nbatch, self.samples - self.silence), dtype=np.float32)
        s = np.zeros((nbatch, self.samples - self.silence), dtype=np.float32)

        for b in range(nbatch):

            if mode0 is None:
                mode = np.random.choice(self.modes, p=p_modes)
            else:
                mode = mode0

            if scenario0 is None:
                scenario = np.random.choice(self.scenarios, p=p_scenarios)
            else:
                scenario = scenario0

            if idx0 is None:
                idx = np.random.choice(self.train_set_length)
            else:
                idx = idx0
            name = f"{self.cache_path}/cache/train/{mode}/{scenario}/{idx:04d}.mat"
            data = load_numpy_from_mat(name)

            x[b, :] = data['x'][0, self.silence:]
            y[b, :] = data['y'][0, self.silence:]
            d[b, :] = data['d'][0, self.silence:]
            e[b, :] = data['e'][0, self.silence:]
            s[b, :] = data['s'][0, self.silence:]

        return x, y, d, e, s

    def load_from_dir(self, name, nbatch):

        data = load_numpy_from_mat(name)

        x = data['x'][0, :]  # shape = (1, samples)
        y = data['y'][0, :]  # shape = (1, samples)
        d = data['d'][0, :]  # shape = (1, samples)
        e = data['e'][0, :]  # shape = (1, samples)

        x = np.stack([x] * nbatch, axis=0)
        y = np.stack([y] * nbatch, axis=0)
        d = np.stack([d] * nbatch, axis=0)
        e = np.stack([e] * nbatch, axis=0)
        s = np.zeros_like(x)

        return x, y, d, e, s

    def load_test(self, idx, nbatch=1):

        name = f"{self.cache_path}/cache/test/{idx:04d}.mat"
        return self.load_from_dir(name, nbatch)

    def load_test_blind(self, idx, nbatch=1):

        name = f"{self.cache_path}/cache/test_blind/{idx:04d}.mat"
        return self.load_from_dir(name, nbatch)

    def write_enhanced(self, x, idx, experiment_name):

        # batchsize = 1
        self.aec_loader.write_enhanced(x[0, :], idx, experiment_name, self.aec_loader.test)

    def write_enhanced_blind(self, x, idx, experiment_name):
        # batchsize = 1
        self.aec_loader.write_enhanced(x[0, :], idx, experiment_name, self.aec_loader.test_blind)

    def write_aec_only(self, ):

        for idx in range(self.test_set_length):
            name = f"{self.cache_path}/cache/test/{idx:04d}.mat"
            data = load_numpy_from_mat(name)
            e = data['e'][0, :]  # shape = (samples,)
            self.aec_loader.write_enhanced(e, idx, 'aec_only')
            print('writing file:', idx, '/', self.test_set_length)

    def write_aec_only_blind(self, ):

        for idx in range(self.blind_test_set_length):
            name = f"{self.cache_path}/cache/test_blind/{idx:04d}.mat"
            data = load_numpy_from_mat(name)
            e = data['e'][0, :]  # shape = (samples,)
            self.aec_loader.write_enhanced_blind(e, idx, 'aec_only')
            print('writing file:', idx, '/', self.blind_test_set_length)
