import argparse
import pickle
import os
import tensorflow as tf
import numpy as np
import time


def __measure_sequential_models(model_loader, input_loader, device, trial):
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device_str = '/CPU:0'
    else:
        device_str = '/GPU:0'
    with tf.device(device_str):
        model = model_loader()
        input_dat = input_loader()
        num_layers = len(model.layers)
        result = np.zeros((num_layers, trial))
        curr_dat = input_dat
        for layer_idx, layer in enumerate(model.layers):
            curr_out = layer(curr_dat)
            for trial_idx in range(trial):
                start_time = time.time()
                layer(curr_dat)
                end_time = time.time()
                result[layer_idx, trial_idx] = end_time - start_time
            curr_dat = curr_out
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('-t', '--trial', type=int)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    model_loader = lambda: pickle.load(open(args.model, "rb"))
    input_loader = lambda: pickle.load(open(args.input, "rb"))

    result = __measure_sequential_models(model_loader, input_loader, args.device, args.trial)
    with open(args.output, "wb") as fout:
        pickle.dump(result, fout)
