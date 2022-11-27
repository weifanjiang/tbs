python3 measurement.py -m examples/cnn-flowers/model.pickle -i examples/cnn-flowers/input.pickle -t 10 -d cpu -o examples/cnn-flowers/measurement_cpu.pickle
python3 measurement.py -m examples/cnn-flowers/model.pickle -i examples/cnn-flowers/input.pickle -t 10 -d gpu -o examples/cnn-flowers/measurement_gpu.pickle
python3 plot_utils.py -m examples/cnn-flowers/model.pickle -c examples/cnn-flowers/measurement_cpu.pickle -g examples/cnn-flowers/measurement_gpu.pickle -o figs/cnn-flowers.pdf

python3 measurement.py -m examples/lm-imdb/model.pickle -i examples/lm-imdb/input.pickle -t 10 -d cpu -o examples/lm-imdb/measurement_cpu.pickle
python3 measurement.py -m examples/lm-imdb/model.pickle -i examples/lm-imdb/input.pickle -t 10 -d gpu -o examples/lm-imdb/measurement_gpu.pickle
python3 plot_utils.py -m examples/lm-imdb/model.pickle -c examples/lm-imdb/measurement_cpu.pickle -g examples/lm-imdb/measurement_gpu.pickle -o figs/lm-imdb.pdf

python3 measurement.py -m examples/rnn-mnist/model.pickle -i examples/rnn-mnist/input.pickle -t 10 -d cpu -o examples/rnn-mnist/measurement_cpu.pickle
python3 measurement.py -m examples/rnn-mnist/model.pickle -i examples/rnn-mnist/input.pickle -t 10 -d gpu -o examples/rnn-mnist/measurement_gpu.pickle
python3 plot_utils.py -m examples/rnn-mnist/model.pickle -c examples/rnn-mnist/measurement_cpu.pickle -g examples/rnn-mnist/measurement_gpu.pickle -o figs/rnn-mnist.pdf
