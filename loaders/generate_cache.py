# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"

import argparse
import math
import sys

import numpy as np
import os

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import mkdir, compensate_delay
from algorithms.ssaec_fast import SSAECFast
from utils.mat_helpers import save_numpy_to_mat
from loaders.aec_loader import AECLoader
from loaders.audio_loader import AudioLoader


class CacheGenerator:

    def __init__(self, dataset_path, cache_path, train_length):
        self.cache_path = cache_path
        self.aec_loader = AECLoader(name='aec_loader', dataset_dir=f"{dataset_path}/aec")
        self.dt_loader = AudioLoader(name='doubletalk', path=f"{dataset_path}/doubletalk")
        self.noise_loader = AudioLoader(name='noise', path=f"{dataset_path}/noise")
        self.noise_loader.cache_files()
        self.ssaec = SSAECFast(wlen=512, tail_length=0.250)

        self.fs = self.aec_loader.fs
        self.samples = self.seconds(15)
        self.silence = self.seconds(5)

        self.dataset_dir = self.aec_loader.dataset_dir
        self.scenarios = ['nearend', 'farend', 'doubletalk']
        self.modes = ['real', 'simu', 'hard']
        self.train_set_length = train_length
        self.test_set_length = len(self.aec_loader.test)
        self.blind_test_set_length = len(self.aec_loader.test_blind)

    def seconds(self, n):
        return int(self.fs * n)

    def repeat_and_trim_files(self, x, d):

        samples = len(x)
        rp = math.ceil(self.samples / samples)
        return np.tile(x, rp)[:self.samples], \
               np.tile(d, rp)[:self.samples]

    def delay_x(self, x):
        samples3 = self.samples // 3
        # create artificial delay changes by shifting the last third randomly
        lag = int(np.random.uniform(-0.020, 0) * self.fs)
        x[2 * samples3:self.samples] = np.roll(x[2 * samples3:self.samples], lag)
        return x

    def attenuate_and_clip_d(self, d):
        samples3 = self.samples // 3
        # randomly attenuate microphone signal in the middle third
        G = np.ones((self.samples,), dtype=np.float32)
        G[samples3:2 * samples3] = np.power(10, np.random.uniform(0, -40) / 20)
        d = d * G

        # randomly add clipping
        G = np.power(10, np.random.uniform(0, +12) / 20)
        d = np.clip(d * G, -1, +1) / G
        return d

    def generate_train(self, mode, scenario, idx):

        x, d = self.aec_loader.load_train(mode, idx)
        x, d = self.repeat_and_trim_files(x, d)

        if mode == 'simu':
            x = self.delay_x(x)
        if mode in ['simu', 'real']:
            d = self.attenuate_and_clip_d(d)

        # load doubletalk
        s = self.dt_loader.load_random_files(samples=self.samples, offset=self.silence)

        # load noise
        n = self.noise_loader.load_random_cached_file(samples=self.samples)

        Pd = 10 * np.log10(np.mean(d ** 2))
        Ps = 10 * np.log10(np.mean(s ** 2))
        Pn = 10 * np.log10(np.mean(n ** 2))
        sir = np.random.uniform(0, +6)  # add doubletalk at 0...+6dB SIR
        snr = np.random.uniform(12, 32)  # add noise at 12...32dB SNR
        s *= np.power(10, (Pd - Ps + sir) / 20)
        n *= np.power(10, (Pd - Pn - snr) / 20)

        # do not add noise to real or hard files
        if mode in ['real', 'hard']:
            n = np.zeros_like(d)

        # simulate scenario
        if scenario == 'nearend':
            x = np.zeros_like(d)

        if scenario == 'farend':
            s = np.zeros_like(d)

        d = d + s + n

        # compensate bulk delay
        x = compensate_delay(x, d)

        # perform EC with random starting position
        start = np.random.choice(self.samples)
        x0 = np.roll(x, start)
        d0 = np.roll(d, start)

        e, y = self.ssaec.run(x0, d0, repeats=2)
        e = np.roll(e, -start)
        y = np.roll(y, -start)

        return x, y, d, e, s

    def generate_test(self, idx, path, dataset):
        x, d = self.aec_loader.load_from(path, dataset, idx)
        x = compensate_delay(x, d)
        e, y = self.ssaec.run(x, d, repeats=2)

        return x, y, d, e

    def cache_train_set(self):

        for mode in self.modes:
            for scenario in self.scenarios:
                for idx in range(self.train_set_length):
                    name = f"{self.cache_path}/cache/train/{mode}/{scenario}/{idx:04d}.mat"

                    x, y, d, e, s = self.generate_train(mode, scenario, idx)
                    data = {
                        'x': x,
                        'y': y,
                        'd': d,
                        'e': e,
                        's': s,
                        'fs': self.fs,
                        'mode': mode,
                        'scenario': scenario,
                    }
                    mkdir(name)
                    save_numpy_to_mat(name, data)
                    print('writing train file:', idx, '/', self.train_set_length)

    def cache_test(self, length, path, dataset):

        for idx in range(length):
            name = f"{self.cache_path}/cache/{path}/{idx:04d}.mat"
            x, y, d, e = self.generate_test(idx, path, dataset)
            data = {
                'x': x,
                'y': y,
                'd': d,
                'e': e,
                'fs': self.fs,
                'idx': idx,
            }
            mkdir(name)
            save_numpy_to_mat(name, data)
            print('writing ' + path + ' file:', idx, '/', length)

    def cache_test_set(self):
        self.cache_test(self.test_set_length, "test", self.aec_loader.test)

    def cache_blind_test_set(self):
        self.cache_test(self.blind_test_set_length, "test_blind", self.aec_loader.test_blind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cache generator for AEC training')
    parser.add_argument('--dataset_path', help='absolute path to dataset',
                        type=str, default='Interspeech_AEC_Challenge_2021')
    parser.add_argument('--cache_path', help='destination to cache',
                        type=str, default='output')
    parser.add_argument('--train_length', help='number of samples for train',
                        type=int, default=10000)
    args = parser.parse_args()

    gc = CacheGenerator(args.dataset_path, args.cache_path, args.train_length)
    gc.cache_train_set()
    gc.cache_test_set()
    gc.cache_blind_test_set()
