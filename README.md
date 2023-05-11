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

A total of `237.212.662` dialogs.
`237.162.662` are used for train split, `25.000` each are used for validation and test splits.

### CommonSense

CommonSense Conversation from [`DiffuSeq`](https://github.com/Shark-NLP/DiffuSeq)
Token statistic collected w/ `facebook/blenderbot-400M-distill` tokenizer, see [`scripts.cc_tokens_stats`](scripts/cc_tokens_stats.py).

**Train**
- 3.382.137 samples
- Context contains 81.772.641 tokens in range 2-84, average 24.178
- Target contains 80.812.361 tokens in range 1-84, average 23.894

**Valid**
- 2.047 samples
- Context contains 49.424 tokens in range 3-53, average 24.133
- Target contains 49.887 tokens in range 2-56, average 24.359

**Test**
- 9.999 samples
- Context contains 241.541 tokens in range 2-58, average 24.154
- Target contains 240.374 tokens in range 2-61, average 24.037

### Fine-tuning

We use the [ConvAI2 Dataset](https://arxiv.org/pdf/1902.00098.pdf) containing dialogues between personas with different descriptive profiles.
The dataset can be downloaded [here](http://parl.ai/downloads/convai2/convai2_fix_723.tgz).
