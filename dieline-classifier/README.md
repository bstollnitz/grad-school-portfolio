# Package dieline classification

*Technologies:* Python, Keras, NumPy, Scikit-learn, Plotly. <br>
*Topics:* deep learning, convolutional neural network (CNN), classification. <br>

## Description

<a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdlejxLv20qqllOVxA?e=jZh3nB">![](docs/poster.png?raw=true)</a>

The goal of this project is to classify all the panels of a package
dieline. This project is a collaboration with Adobe Research, and will
allow researchers to build software that automatically simulates the
folding of a 2D dieline into a 3D model of a package.

I collaborated with Adobe to define two datasets
for this project. In the first dataset, each panel of a dieline is
represented using a feature vector of integers expressing the
number of occurrences of each angle in the panel outline. 
In the second dataset, each panel is
represented as an image that includes a bit of the
surrounding area.

I automated the process of hyperparameter tuning by
performing grid search over a random sample of the full
combination of hyperparameters. I ran two rounds of
hyperparameter tuning for each dataset type. The first round had
a large search space, including hyperparameters like learning
rate, batch size, and optimizer settings, as well model selection
parameters for the network. The second round relied on the
results from the first round to reduce the search space to the
most promising values. This two-step approach helped increase
the accuracy of the results.

I achieved the highest test accuracy for the image dataset (using a
CNN with two convolution layers). However, the vector dataset has
several advantages: the data is much more compact and the
associated network is smaller, and therefore, both training and
prediction can be done far more quickly.

You can find more details in the 
<a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdkoiZzuMIJLGgFG2Q?e=9sUqhs">report</a> 
and <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdlejxLv20qqllOVxA?e=jZh3nB">poster</a> 
for this project.

This was my final project for the Deep Learning class (CSE 599) at the 
University of Washington, which I completed as part of my masters in 
Applied Mathematics.

## Running

The dataset used in this project is the property of Adobe and can not be shared. 
Therefore, you're welcome to browse the code but you'll have a hard time
running it.

## Development

If you want to look at the code, it's in the `dieline_classifier` directory. Start 
with `main.py`.

Here's a quick description of the directories in this repo:

- `data` - Empty.
- `dieline_classifier` - The code. 
- `graphs` - Histogram with the categories of the input data.
- `tests` - Tests for custom hyperparameter tuner.
- `vector_round1` - Results from first round of hyperparameter tuning on vector data.
- `vector_round2` - Results from second round of hyperparameter tuning on vector data.
- `image_round1` - Results from first round of hyperparameter tuning on image data.
- `image_round2` - Results from second round of hyperparameter tuning on image data.

The `environment.yml` file contains the Conda environment used to set up the 
prerequisites to run the code.