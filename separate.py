#!/usr/bin/env python

import argparse
import os
import soundfile as sf
import librosa
import torch
import threading

from data import EvalDataLoader, EvalDataset
from net import TasNet
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED

parser = argparse.ArgumentParser('Separate speech using TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')

parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('--thread_num', default=2, type=int,
                    help='Num of Threads')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample Rate')
parser.add_argument('--process_num', default=4, type=int,
                    help='Num of Process')

def inference(args,model,data):
    mixture,mix_lengths,filenames = data
    if(args.use_cuda):
        mixture = mixture.cuda()
    with torch.no_grad():
        estimate_source = model(mixture, mix_lengths)
        flat_estimate = remove_pad_and_flat(estimate_source, mix_lengths)
        mixture = remove_pad_and_flat(mixture, mix_lengths)
        # Write result
        def write(inputs, filename, sr=args.sample_rate):
            sf.write(filename, inputs, sr)
        for i, filename in enumerate(filenames):
            filename = os.path.join(args.out_dir,
                                    os.path.basename(filename).strip('.wav'))
            write(mixture[i], filename + '.wav')
            C = flat_estimate[i].shape[0]
            for c in range(C):
                write(flat_estimate[i][c], filename + '_s{}.wav'.format(c+1))


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = TasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate, L=model.L)
    eval_loader =  EvalDataLoader(eval_dataset, batch_size=1,num_workers=args.process_num)
    os.makedirs(args.out_dir, exist_ok=True)

    threads = []
    for data in eval_loader:
        thread = threading.Thread(target = inference,args=(model,data,args))
        threads.append(thread)
    max_threads = args.thread_nums

    for i, thread in enumerate(threads):
        thread.start()
    # 每当达到最大线程数时，等待一个线程结束后再启动下一个线程
        if (i + 1) % max_threads == 0:
            thread.join()
        
def remove_pad_and_flat(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, K, L] or [B, K, L]
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T]
    """


    results = []
    dim = inputs.dim()
    if dim == 4:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 4: # [B, C, K, L]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 3:  # [B, K, L]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)
