from setuptools import setup, find_packages

setup(
    name="src",
    version="1.0",
    description="Diffusion src",
    author="rrevoid",
    packages=find_packages(),
    scripts=[
        "didi/scripts/evaluate_model.py",
        "didi/scripts/train_model.py",
    ],
)
