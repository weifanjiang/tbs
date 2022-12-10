#!/bin/bash

python3 measurement.py -p examples/cnn-flowers -d cpu -b 64 -t 10
python3 measurement.py -p examples/cnn-flowers -d cpu -b 128 -t 10
python3 measurement.py -p examples/cnn-flowers -d cpu -b 256 -t 10
python3 measurement.py -p examples/cnn-flowers -d cpu -b 512 -t 10
python3 measurement.py -p examples/cnn-flowers -d cpu -b 1024 -t 10

python3 measurement.py -p examples/cnn-flowers -d gpu -b 64 -t 10
python3 measurement.py -p examples/cnn-flowers -d gpu -b 128 -t 10
python3 measurement.py -p examples/cnn-flowers -d gpu -b 256 -t 10
python3 measurement.py -p examples/cnn-flowers -d gpu -b 512 -t 10
python3 measurement.py -p examples/cnn-flowers -d gpu -b 1024 -t 10

python3 measurement.py -p examples/lm-imdb -d cpu -b 64 -t 10
python3 measurement.py -p examples/lm-imdb -d cpu -b 128 -t 10
python3 measurement.py -p examples/lm-imdb -d cpu -b 256 -t 10
python3 measurement.py -p examples/lm-imdb -d cpu -b 512 -t 10
python3 measurement.py -p examples/lm-imdb -d cpu -b 1024 -t 10

python3 measurement.py -p examples/lm-imdb -d gpu -b 64 -t 10
python3 measurement.py -p examples/lm-imdb -d gpu -b 128 -t 10
python3 measurement.py -p examples/lm-imdb -d gpu -b 256 -t 10
python3 measurement.py -p examples/lm-imdb -d gpu -b 512 -t 10
python3 measurement.py -p examples/lm-imdb -d gpu -b 1024 -t 10

python3 measurement.py -p examples/rnn-mnist -d cpu -b 64 -t 10
python3 measurement.py -p examples/rnn-mnist -d cpu -b 128 -t 10
python3 measurement.py -p examples/rnn-mnist -d cpu -b 256 -t 10
python3 measurement.py -p examples/rnn-mnist -d cpu -b 512 -t 10
python3 measurement.py -p examples/rnn-mnist -d cpu -b 1024 -t 10

python3 measurement.py -p examples/rnn-mnist -d gpu -b 64 -t 10
python3 measurement.py -p examples/rnn-mnist -d gpu -b 128 -t 10
python3 measurement.py -p examples/rnn-mnist -d gpu -b 256 -t 10
python3 measurement.py -p examples/rnn-mnist -d gpu -b 512 -t 10
python3 measurement.py -p examples/rnn-mnist -d gpu -b 1024 -t 10
