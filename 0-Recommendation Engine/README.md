# Welcome to Recommendation Engine

This is a repository that consists of the code and the model-web API of 
a product recommendation engine of an online retailer.

## Getting started

Set up the conda environment from your command line. If you're working locally,
[Anaconda Prompt](https://docs.anaconda.com/anaconda/install/) is recommended for doing this. If you're
working on a cluster, then the default command line should be sufficient. We suggest 
setting up Python `3.7` within the conda environment. From hereon, we refer to the 
custom conda environment as `recommen`:

```
conda create -n recommen python=3.7
```

Afterwards, activate the conda environment and `pip` install the package, and the 
dependencies as listed in the `requirements.txt` file:

```
conda activate recommen
pip install -e .
pip install -r requirements.txt
```

This may take a few minutes. Afterwards, feel free to start up a Jupyter Notebook or run web API and
play around with the code!


User interface of the app is quite simple. It is possible to use css files or different html components to improve it.

To use web-API please run app.py file. It is also available in Recommendation_Engine.ipynb file.

