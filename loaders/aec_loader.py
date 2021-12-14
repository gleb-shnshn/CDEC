# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"

import os
import sys

import numpy as np

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import mstft, mistft, apply_lowpass_filter, apply_highpass_filter, audioread, mkdir, \
    audiowrite


class AECLoader:

    def __init__(self, name='aec_loader', dataset_dir='../../Interspeech_AEC_Challenge_2021/AEC-Challenge/datasets',
                 fs=16e3):

        self.fs = fs
        self.name = name
        self.dataset_dir = dataset_dir

        # load train_real files
        self.train_real = self.get_content_of_dir(dataset_dir, "real")

        # load train_simu files
        self.train_simu = self.get_content_of_dir(dataset_dir, "simu")

        # load train_hard files
        self.train_hard = self.get_content_of_dir(dataset_dir, "hard")

        # load test files
        self.test = self.get_content_of_dir(dataset_dir, "test")

        # load test_blind files
        self.test_blind = self.get_content_of_dir(dataset_dir, "test_blind")

        print('*** audio loader "', self.name, '" found', len(self.train_real), 'train_real files')
        print('*** audio loader "', self.name, '" found', len(self.train_simu), 'train_simu files')
        print('*** audio loader "', self.name, '" found', len(self.train_hard), 'train_hard files')
        print('*** audio loader "', self.name, '" found', len(self.test), 'test files')
        print('*** audio loader "', self.name, '" found', len(self.test_blind), 'test_blind files')

    def get_content_of_dir(self, dataset_dir, name):
        return sorted(list(set(os.listdir(f"{dataset_dir}/{name}/mic"))
                           .intersection(set(os.listdir(f"{dataset_dir}/{name}/lpb")))))

    def hp_filter(self, x):

        Fx = mstft(x)
        Fx = apply_highpass_filter(Fx, self.fs, fc=100, order=4)
        x = mistft(Fx)

        return x

    def lp_filter(self, x):

        Fx = mstft(x)
        Fx = apply_lowpass_filter(Fx, self.fs, fc=7500, order=8)
        x = mistft(Fx)

        return x

    def load_train(self, mode, idx=None):

        if mode == 'real':
            return self.load_train_dataset("real", self.train_real, idx)

        elif mode == 'simu':
            return self.load_train_dataset("simu", self.train_simu, idx)

        elif mode == 'hard':
            return self.load_train_dataset("hard", self.train_hard, idx)

    def get_audio(self, name, dataset, idx):
        x, fs = audioread(f"{self.dataset_dir}/{name}/lpb/{dataset[idx]}")
        d, fs = audioread(f"{self.dataset_dir}/{name}/mic/{dataset[idx]}")
        return x, fs, d, fs

    def load_train_dataset(self, name, dataset, idx=None):
        if idx is not None:
            idx = np.mod(idx, len(dataset))
        else:
            idx = np.random.choice(len(dataset))

        return self.load_from(name, dataset, idx)

    def load_from(self, name, dataset, idx):
        x, fs, d, fs = self.get_audio(name, dataset, idx)

        x = self.hp_filter(x)
        d = self.hp_filter(d)
        d = self.lp_filter(d)

        samples = min(len(x), len(d))
        x = x[:samples]
        d = d[:samples]

        Pd = 10 * np.log10(np.mean(d ** 2))
        G = np.power(10, (-26 - Pd) / 20)
        x *= G
        d *= G
        return x, d

    def write_enhanced(self, x, idx, subfolder, dataset):

        G = 0.99 / np.max(np.abs(x))
        x *= np.minimum(G, 1)

        name = f"{self.dataset_dir}/submission/{subfolder}/{dataset[idx].replace('.wav', '_enh.wav')}"
        mkdir(name)
        audiowrite(x, name)

