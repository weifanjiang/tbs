import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-r', '--redraw', action='store_true')
    parser.add_argument('-c', '--cost', action='store_true')
    args = parser.parse_args()

    devices = ['cpu', 'gpu']
    batch_sizes = [64, 128, 256, 512, 1024]  # batch size used for measurement
    path, redraw = args.path, args.redraw
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
        fig_save_path = os.path.join(fig_dir, '{}_batchSize{}.pdf'.format(basename, batch_size))
        if redraw or (not os.path.isfile(fig_save_path)):
            fig, ax = plt.subplots(1, 1)
            for device in devices:
                mat = measurement_results[(device, batch_size)]
                median_time = np.median(mat, axis=1)
                if device == 'cpu':
                    pos = [i - bar_width/2 for i in x_vals]
                else:
                    pos = [i + bar_width/2 for i in x_vals]
                percentiles = np.percentile(a=mat, q=[25, 75], axis=1)
                ax.bar(x=pos, height=median_time, label=device, width=bar_width, yerr=percentiles, color=color_dict[device])
            ax.legend(title='device', bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.set_yscale('log')

            if not args.cost:
                ax.set_ylabel('time per sample (log seconds)')
            else:
                ax.set_ylabel('cost per sample (log cost per second)')

            ax.set_title('{} (batch size {})'.format(basename, batch_size))
            ax.set_xticks(x_vals)
            ax.set_xticklabels(layer_names, rotation=90)
            plt.savefig(fig_save_path, bbox_inches='tight')

    # draw per-layer runtime w.r.t. batch size
    x_vals = [i for i in range(1, len(batch_sizes) + 1)]
    for layer_idx, layer_name in enumerate(layer_names):
        fig_save_path = os.path.join(fig_dir, '{}.pdf'.format(layer_name))
        if redraw or (not os.path.isfile(fig_save_path)):
            fig, ax = plt.subplots(1, 1)
            for device in devices:
                avg_time, lower, upper = list(), list(), list()
                for batch_size in batch_sizes:
                    layer_runtimes = measurement_results[(device, batch_size)][layer_idx, :]
                    avg_time.append(np.median(layer_runtimes))
                    lower.append(np.percentile(a=layer_runtimes, q=25))
                    upper.append(np.percentile(a=layer_runtimes, q=75))
                ax.fill_between(x_vals, y1=lower, y2=upper, color=color_dict[device], alpha=0.3)
                ax.plot(x_vals, avg_time, label=device, c=color_dict[device])
            ax.legend(title='device', bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.set_yscale('log')

            if not args.cost:
                ax.set_ylabel('time per sample (log seconds)')
            else:
                ax.set_ylabel('cost per sample (log cost per second)')

            ax.set_xlabel('batch size')
            ax.set_title('{} ({})'.format(layer_name, basename))
            ax.set_xticks(x_vals)
            ax.set_xticklabels(batch_sizes)
            plt.savefig(fig_save_path, bbox_inches='tight')
