# setupfile

from setuptools import setup, find_packages

setup(
    name="planparse",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "accelerate >= 1.1.0",
        "evaluate >= 0.4.3",
        "Jinja2 >= 3.1.4",
        "numpy >= 2.0.2",
        "pandas >= 2.2.3",
        "peft >= 0.13.2",
        "scikit-learn >= 1.5.2",
        "torch >= 2.5.0",
        "tqdm >= 4.66.5",
        "transformers >= 4.47.0",
        "bitsandbytes >= 0.44.1",
    ],
    author="Steffen Magnussen & Eirik Vinje",
    author_email="eirik.matias@gmail.com & steffenmag",
    description="A project to parse plan data",
    keywords="regulations, plans, data",
)