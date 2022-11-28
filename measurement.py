import argparse
import pickle
import os
import tensorflow as tf
import numpy as np
import time


def __measure_sequential_models(model_loader, input_loader, device, trial, batch_size):
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device_str = '/CPU:0'
    else:
        device_str = '/GPU:0'
    with tf.device(device_str):
        model = model_loader()
        input_dat = input_loader()
        num_samples = input_dat.shape[0]
        num_layers = len(model.layers)
        result = np.zeros((num_layers, trial))
        curr_dat = input_dat
        for layer_idx, layer in enumerate(model.layers):
            print('Measuring layer {}'.format(layer.name))
            curr_out = layer(curr_dat)
            curr_idx = 0
            for trial_idx in range(trial):
                print('trial {}/{}'.format(trial_idx + 1, trial))
                start_time = time.time()
                curr_idx = 0
                while curr_idx < num_samples:
                    end_idx = min(curr_idx + batch_size, num_samples)
                    layer(curr_dat[curr_idx:end_idx])
                    curr_idx = end_idx
                end_time = time.time()
                result[layer_idx, trial_idx] = (end_time - start_time)/num_samples
            curr_dat = curr_out
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('-t', '--trial', type=int)
    args = parser.parse_args()

    model_path = os.path.join(args.path, 'model.pickle')
    input_path = os.path.join(args.path, 'input.pickle')

    if (os.path.isfile(model_path) == False) or (os.path.isfile(input_path) == False):
        os.system('python3 {}'.format(os.path.join(args.path, 'gen.py')))

    model_loader = lambda: pickle.load(open(model_path, "rb"))
    input_loader = lambda: pickle.load(open(input_path, "rb"))

    result = __measure_sequential_models(model_loader, input_loader, args.device, args.trial, args.batch)
    out_name = os.path.join(args.path, "{}-{}.pickle".format(args.device, args.batch))
    with open(out_name, "wb") as fout:
        pickle.dump(result, fout)
