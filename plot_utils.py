import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()

    # batch size used for measurement
    devices = ['cpu', 'gpu']
    batch_sizes = [64, 128, 256, 512, 1024]
    path = args.path
    basename = os.path.basename(path)
    fig_dir = os.path.join('figs', basename)
    os.system('mkdir -p {}'.format(fig_dir))
    model_path = os.path.join(path, 'model.pickle')
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    
    layer_names = [x.name for x in model.layers]
    measurement_results = dict()  # key: (device, batch size)
    for device in devices:
        for batch_size in batch_sizes:
            with open(os.path.join(path, '{}-{}.pickle'.format(device, batch_size)), 'rb') as fin:
                measurement_results[(device, batch_size)] = pickle.load(fin)
    
    # draw bar plots
    color_dict = {
        'cpu': 'tab:blue',
        'gpu': 'tab:orange'
    }

    x_vals = [i for i in range(1, len(layer_names) + 1)]
    bar_width = 0.3
    for batch_size in batch_sizes:
        fig, ax = plt.subplots(1, 1)
        for device in devices:
            mat = measurement_results[(device, batch_size)]
            avg_time = np.average(mat, axis=1)
            if device == 'cpu':
                pos = [i - bar_width/2 for i in x_vals]
            else:
                pos = [i + bar_width/2 for i in x_vals]
            percentiles = np.percentile(a=mat, q=[25, 75], axis=1)
            ax.bar(x=pos, height=avg_time, label=device, width=bar_width, yerr=percentiles, color=color_dict[device])
        ax.legend(title='device', loc='upper right')
        ax.set_yscale('log')
        ax.set_ylabel('time per sample (log seconds)')
        ax.set_title('{} (batch size {})'.format(basename, batch_size))
        ax.set_xticks(x_vals)
        ax.set_xticklabels(layer_names, rotation=90)
        plt.savefig(os.path.join(fig_dir, '{}_batchSize{}.pdf'.format(basename, batch_size)), bbox_inches='tight')
