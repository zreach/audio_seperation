import os
import numpy as np
import soundfile as sf
from activlev import activlev
import argparse
import tqdm
import logging

# def cut(array,t):
#     if(len(array)>t):
#         return 
def GenerateMixAudio(dataPath, state, useActive=True):
    if state.upper() == 'TRAIN':
        dataType = ['tr', 'cv']
    else:
        dataType = ['tt']
    print(dataType)
    for i_type in dataType:
        audio_path = os.path.join(dataPath, 'audio', i_type)
        if not os.path.exists(os.path.join(audio_path)):
            os.makedirs(os.path.join(audio_path))

        outS1 = os.path.join(audio_path, 's1')
        outS2 = os.path.join(audio_path, 's2')
        outMix = os.path.join(audio_path, 'mix')

        if not os.path.exists(outS1):
            os.mkdir(outS1)
        if not os.path.exists(outS2):
            os.mkdir(outS2)
        if not os.path.exists(outMix):
            os.mkdir(outMix)



        taskFile = os.path.join(dataPath, 'mix_files', "{}.txt".format(i_type))
	
    
        if not os.path.exists(os.path.join(dataPath, 'text')):
            os.mkdir(os.path.join(dataPath, 'text'))
        sourceFile1 = os.path.join(dataPath, 'text', "{}_1".format(i_type))
        sourceFile2 = os.path.join(dataPath, 'text', "{}_2".format(i_type))
        mixFile = os.path.join(dataPath, 'text', "{}_mix".format(i_type))

        f1 = open(sourceFile1, 'w')
        f2 = open(sourceFile2, 'w')
        f3 = open(mixFile, 'w')
        logging.info("Processing {}".format(i_type))

        with open(taskFile, 'r') as f:
            for line in tqdm.tqdm(f.readlines()):
                line = line.split()

                s1_tr = line[0]
                s2_tr = line[2]
                s1WavName = "{}_{}".format(line[0].split('/')[-2], line[0].split('/')[-1][:-4])
                s2WavName = "{}_{}".format(line[2].split('/')[-2], line[2].split('/')[-1][:-4])
                s1Snr = round(float(line[1]), 4)
                s2Snr = round(float(line[-1]), 4)
                mixName = "{}_{}_{}_{}".format(s1WavName, s1Snr, s2WavName, s2Snr)

                f1.write(s1_tr)
                f1.write('\n')
                f2.write(s2_tr)
                f2.write('\n')
                f3.write(mixName)
                f3.write('\n')

                s1_16k, fs = sf.read(line[0])
                s2_16k, _ = sf.read(line[2])
                '''
                 In original create_mixtures.m, activlev must be done, which I think it may degrade the performance since it nonlinearly filters the signal
                 However, most of experiments did that parts because this it's essential to control variable for publishing papers.
                '''
                if useActive:
                    s1_16k, lev1 = activlev(s1_16k, fs, 'n')
                    s2_16k, lev2 = activlev(s2_16k, fs, 'n')

                weight_1 = pow(10, s1Snr / 20)
                weight_2 = pow(10, s2Snr / 20)

                s1_16k = weight_1 * s1_16k
                s2_16k = weight_2 * s2_16k

                mix_16k_length = min(len(s1_16k), len(s2_16k))
                s1_16k = s1_16k[:mix_16k_length]
                s2_16k = s2_16k[:mix_16k_length]

                mix_16k = s1_16k + s2_16k
                max_amp_16k = max(np.concatenate(( np.abs(mix_16k), np.abs(s1_16k), np.abs(s2_16k))))
                mix_scaling_16k = 1 / max_amp_16k * 0.9
                s1_16k = mix_scaling_16k * s1_16k
                s2_16k = mix_scaling_16k * s2_16k
                mix_16k = mix_scaling_16k * mix_16k
                threshold = 3 * fs
                s1_16k = s1_16k[:threshold]
                s2_16k = s2_16k[:threshold]
                mix_16k = mix_16k[:threshold]
                
                s1_out = os.path.join(outS1, "{}.wav".format(mixName))
                s2_out = os.path.join(outS2, "{}.wav".format(mixName))
                mix_out = os.path.join(outMix, "{}.wav".format(mixName))
                sf.write(s1_out, s1_16k, fs, format='WAV', subtype='PCM_16')
                sf.write(s2_out, s2_16k, fs, format='WAV', subtype='PCM_16')
                sf.write(mix_out, mix_16k, fs, format='WAV', subtype='PCM_16')

def main(args):
    logging.basicConfig(level=logging.INFO)

    dataPath = args.data_dir
    state = args.state
    useActive = args.use_active
    GenerateMixAudio(dataPath, state, useActive)
    logging.info("Finish generating mixture audio and mixture files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command to generate mixture audio and mixutre files'
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help='Input mixtures sources information data_dir as well as output data directory'
    )
    parser.add_argument(
        "--state",
        type=str,
        help='Define Generating train or test data '
    )
    parser.add_argument(
        "--use_active",
        type=str,
        help='Measure active speech level '
    )
    args = parser.parse_args()
    main(args)
