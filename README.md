# 💬 DiDi: Dialog Diffusion model


[![Main](https://github.com/deepvk/didi/actions/workflows/main.yaml/badge.svg)](https://github.com/deepvk/didi/actions/workflows/main.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

*Work in progress.*

## Structure

- [`src`](./src) ‒ main source code with model and dataset implementations and code to train, test or infer model.
- [`notebooks`](./notebooks) ‒ notebooks with experiments and visualizations.
- [`scripts`](./scripts) ‒ different useful scripts, e.g. print dataset examples or evaluate existing models.
- [`tests`](./tests) ‒ unit tests.

## Requirements

Create virtual environment with `venv` or `conda` and install requirements:
```bash
pip install -r requirements.txt
```

For proper contributions, also use dev requirements:
```bash
pip install -r requirements-dev.txt
```

## Data

### Pretrain

We use [`pushshift.io`](https://pushshift.io/) dataset with Reddits' comments to pretrain our model.
We have collected all the comments for 2019.

TODO: add preprocessing and filter steps

A total of 237.212.662 dialogs, 100.000 each, are used for validation and testing.

### Fine-tuning

We use the [ConvAI2 Dataset](https://arxiv.org/pdf/1902.00098.pdf) containing dialogues between personas with different descriptive profiles.
The dataset can be downloaded [here](http://parl.ai/downloads/convai2/convai2_fix_723.tgz).
