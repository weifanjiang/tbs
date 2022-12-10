# TensorFlow Benchmarking System

## Usage

To benchmark a model and dataset combination, include a `gen.py` file that will produce `model.pickle` and `input.pickle`.

The measurement is done by running:
```
python3 measurement.py [-p PATH] [-d {cpu,gpu}] [-b BATCH] [-t TRIAL] [-c COST]
```

which `-p` is the directory path that contains the `gen.py` file, `-d` is the hardware device to run the model on, `-b` is batch size for inference, and `-t` is the number of trials for the measurement. `-c` is an optional argument; it allows the user to specify an approximation for the hourly rate of the underlying architecture.

Please see `run_exp.sh` for example usages on three sample models we include: CNN, language model, and RNN.
