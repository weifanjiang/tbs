import matplotlib.pyplot as plt
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-c', '--cpu', type=str)
    parser.add_argument('-g', '--gpu', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    with open(args.model, "rb") as fin:
        model = pickle.load(fin)
    
    with open(args.cpu, "rb") as fin:
        cpu_result = pickle.load(fin)
    
    with open(args.gpu, "rb") as fin:
        gpu_result = pickle.load(fin)
    
    layer_names = [x.name for x in model.layers]
    
    assert(len(layer_names) == cpu_result.shape[0])
    assert(len(layer_names) == gpu_result.shape[0])
    
    layer_num = len(layer_names)
    x_vals = range(1, layer_num + 1)
    cpu_x_vals = [x - 0.2 for x in x_vals]
    gpu_x_vals = [x + 0.2 for x in x_vals]

    fig, ax = plt.subplots(1, 1)

    bp1 = ax.boxplot(cpu_result.transpose(), positions=cpu_x_vals, widths=0.3, boxprops=dict(color='blue'))
    bp2 = ax.boxplot(gpu_result.transpose(), positions=gpu_x_vals, widths=0.3, boxprops=dict(color='tab:brown'))

    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ['cpu', 'gpu'],
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(layer_names, rotation=90)
    ax.set_ylabel('time (seconds)')
    plt.savefig(args.output, bbox_inches='tight')
