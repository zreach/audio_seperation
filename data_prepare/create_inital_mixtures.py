import os
import random
import numpy as np
import argparse
import logging
import soundfile as sf
def CreateFiles(input_dir, output_dir, nums_file, state):

    mix_files = os.path.join(output_dir, 'mix_files')
    if not os.path.exists(mix_files):
        os.mkdir(mix_files)
    
    a_wavList = []
    b_wavList = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            

            if (file.endswith('WAV') or file.endswith('wav') or file.endswith('flac')):
                # print('true',end='\r')
                wavFile = os.path.join(root, file)
                data, sr = sf.read(wavFile)
                if len(data.shape) != 1:
                    raise ValueError
                # print(data.shape)
                if data.shape[0] < sr * 3:
                    pass
                else:
                    if('/A/' in root):
                        a_wavList.append(wavFile)
                    elif('/B/' in root):
                        b_wavList.append(wavFile)
                # print('false',end='\r')
                # print('false')
    random.shuffle(a_wavList)
    random.shuffle(b_wavList)
    print(len(a_wavList))
    print(len(b_wavList))
    mix_list = []

    for i in range(len(a_wavList)):
        for j in range(len(b_wavList)):
            mix_list.append([a_wavList[i],b_wavList[j]])
    random.shuffle(mix_list)
    print(len(mix_list))
    mix_list = mix_list[:nums_file]
    mix_list_tr = mix_list[:len(mix_list)-int(len(mix_list)*0.2)]
    mix_list_cv = mix_list[len(mix_list)-int(len(mix_list)*0.2):len(mix_list)-int(len(mix_list)*0.1)]
    mix_list_tt = mix_list[len(mix_list)-int(len(mix_list)*0.1):]
    tr_file = os.path.join(mix_files, 'tr.txt')
    cv_file = os.path.join(mix_files, 'cv.txt')
    tt_file = os.path.join(mix_files, 'tt.txt')
    with open(tr_file, 'w') as ftr:
        for mix in  mix_list_tr:

            snr = np.random.uniform(0, 2.5)
            line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
            ftr.write(line)
        ftr.close()
    with open(cv_file, 'w') as ftr:
        for mix in  mix_list_cv:

            snr = np.random.uniform(0, 2.5)
            line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
            ftr.write(line)
        ftr.close()
    with open(tt_file, 'w') as ftr:
        for mix in  mix_list_tt:

            snr = np.random.uniform(0, 2.5)
            line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
            ftr.write(line)
        ftr.close()



def run(args):
    logging.basicConfig(level=logging.INFO)

    input_dir =args.input_dir
    output_dir = args.output_dir
    state = args.state
    nums_file = args.nums_files
    CreateFiles(input_dir, output_dir, nums_file, state)
    logging.info("Done create initial data pair")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command to make separation dataset'
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to input data directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--nums_files",
        type=int,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--state",
        type=str,
        help='Whether create train or test data directory'
    )
    args = parser.parse_args()
    run(args)
