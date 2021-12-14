import glob

import os
import shutil

AEC_PATH = "../../Interspeech_AEC_Challenge_2021/AEC-Challenge/datasets"
DESTINATION = "../../datasets"


def convert_aec(mode, glob_lpb, glob_mic, replace_lpb, replace_mic):
    lpb = glob.glob(glob_lpb)
    mic = glob.glob(glob_mic)

    print(f"MODE: {mode}, LPB: {len(lpb)} files, MIC: {len(mic)} files")

    os.makedirs(f"{DESTINATION}/{mode}", exist_ok=True)
    os.makedirs(f"{DESTINATION}/{mode}/lpb", exist_ok=True)
    os.makedirs(f"{DESTINATION}/{mode}/mic", exist_ok=True)

    for file in lpb:
        shutil.copyfile(file, f"{DESTINATION}/{mode}/lpb/{file.split('/')[-1].replace(replace_lpb, '')}")

    for file in mic:
        shutil.copyfile(file, f"{DESTINATION}/{mode}/mic/{file.split('/')[-1].replace(replace_mic, '')}")


def convert_audio(mode, glob_audio):
    files = glob.glob(glob_audio)

    os.makedirs(f"{DESTINATION}/{mode}", exist_ok=True)

    for file in files:
        shutil.copyfile(file, f"{DESTINATION}/{mode}/{file.split('/')[-1]}")


convert_aec("real",
            f"{AEC_PATH}/real/*_farend_singletalk_*lpb.wav",
            f"{AEC_PATH}/real/*_farend_singletalk_*mic.wav",
            '_lpb',
            '_mic')

convert_aec("simu",
            f"{AEC_PATH}/synthetic/farend_speech/farend_speech_fileid_*.wav",
            f"{AEC_PATH}/synthetic/echo_signal/echo_fileid_*.wav",
            'farend_speech_',
            'echo_')

convert_aec("hard",
            f"{AEC_PATH}/train_hard/*/*lpb.wav",
            f"{AEC_PATH}/train_hard/*/*mic.wav",
            '_lpb',
            '_mic')

convert_aec("test",
            f"{AEC_PATH}/test_set_interspeech2021/*/*lpb.wav",
            f"{AEC_PATH}/test_set_interspeech2021/*/*mic.wav",
            '_lpb',
            '_mic')

convert_aec("test_blind",
            f"{AEC_PATH}/blind_test_set_interspeech2021/*/*lpb.wav",
            f"{AEC_PATH}/blind_test_set_interspeech2021/*/*mic.wav",
            '_lpb',
            '_mic')

convert_audio("noise", "../../youtube_noise/*/")

convert_audio("doubletalk", "../../wsj0/si_tr_*/*/")
