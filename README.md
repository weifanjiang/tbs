# TensorFlow Benchmarking System

## Measure sequential model runtime

**CNN on flower dataset**
```bash
python3 examples/cnn-flowers/gen.py
python3 measurement.py -m examples/cnn-flowers/model.pickle -i examples/cnn-flowers/input.pickle -t 10 -d cpu -o examples/cnn-flowers/measurement_cpu.pickle
python3 measurement.py -m examples/cnn-flowers/model.pickle -i examples/cnn-flowers/input.pickle -t 10 -d gpu -o examples/cnn-flowers/measurement_gpu.pickle
```

**Language modeling**
```bash
python3 examples/lm-imdb/gen.py
python3 measurement.py -m examples/lm-imdb/model.pickle -i examples/lm-imdb/input.pickle -t 10 -d cpu -o examples/lm-imdb/measurement_cpu.pickle
python3 measurement.py -m examples/lm-imdb/model.pickle -i examples/lm-imdb/input.pickle -t 10 -d gpu -o examples/lm-imdb/measurement_gpu.pickle
```

**RNN**
```bash
python3 examples/rnn-mnist/gen.py
python3 measurement.py -m examples/rnn-mnist/model.pickle -i examples/rnn-mnist/input.pickle -t 10 -d cpu -o examples/rnn-mnist/measurement_cpu.pickle
python3 measurement.py -m examples/rnn-mnist/model.pickle -i examples/rnn-mnist/input.pickle -t 10 -d gpu -o examples/rnn-mnist/measurement_gpu.pickle
```
