import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from data import AudioDataLoader, AudioDataset
from criterion import cal_loss
from net import TasNet

parser = argparse.ArgumentParser('Evaluate separation performance using TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = TasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, L=model.L)
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            # Forward
            estimate_source = model(padded_mixture, mixture_lengths)  # [B, C, K, L]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            mixture = remove_pad_and_flat(padded_mixture, mixture_lengths)
            source = remove_pad_and_flat(padded_source, mixture_lengths)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad_and_flat(reorder_estimate_source,
                                                  mixture_lengths)
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))

# def cal_snr(signal, reference):
#     # 计算信号的能量
#     signal_energy = np.sum(signal ** 2)
#     reference_energy = np.sum(reference ** 2)

#     # 计算SDR
#     sdr = 10 * np.log10(signal_energy / reference_energy)

#     return sdr
def cal_SISNRi(src_ref, src_est, mix):
    """计算SISNRi

    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    sdr0 = []
    sdr = []
    # print(src_ref)
    all_sdri = 0
    # print(src_ref.size())
    C = 2 

    for i in range(C):
        sdr.append(cal_SISNR(src_ref[i],src_est[i]))
        sdr0.append(cal_SISNR(src_ref[i],mix))
        all_sdri += sdr[i] - sdr0[i]
    avg_SDRi = all_sdri / C
    return avg_SDRi

def cal_SISNR(ref_signal,ori_signal,EPS=1e-8):
    noise = ref_signal-ori_signal

    ref_signal = ref_signal - np.mean(ref_signal)
    ori_signal = ori_signal - np.mean(ori_signal)

    noise_energy = np.sum(noise**2) + EPS
    target_energy = np.sum(ori_signal**2) + EPS
    signal_T = np.sum(ref_signal * ori_signal) * ori_signal / target_energy
    T_energy = np.sum(signal_T**2) + EPS
    return 10 * np.log10(T_energy/noise_energy)

def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


            
def remove_pad_and_flat(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, K, L] or [B, K, L]
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
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
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    args = parser.parse_args()
    print(args)
    evaluate(args)
