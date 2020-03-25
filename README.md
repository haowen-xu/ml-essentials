# ML Essentials

[![](https://github.com/haowen-xu/ml-essentials/workflows/unittest/badge.svg?branch=master)](https://github.com/haowen-xu/ml-essentials/actions)
[![](https://coveralls.io/repos/github/haowen-xu/ml-essentials/badge.svg?branch=master)](https://coveralls.io/github/haowen-xu/ml-essentials?branch=master)

A set of essential toolkits for daily machine learning experiments.

## Requirements

* Python >= 3.6

## Installation

```bash
pip install git+https://github.com/haowen-xu/ml-essentials.git
``` 

## Tutorials

### MLRunner

You may run an experiment with the MLRunner, such that its information and output
can be saved via [MLStorage Server](https://github.com/haowen-xu/mlstorage-server).

```bash
mlrun -s http://server:port -- python train.py
```

