# Bea Stollnitz projects

This repository contains machine learning and data science related projects completed by Bea Stollnitz. All code is in Python.

- **Package dieline classification** <br>
  *Technologies:* Python, Keras, NumPy, Scikit-learn, Plotly. <br>
  *Topics:* Fully connected network, CNN, RNN, automated hyperparameter tuning, classification. <br>

- **Predicting time evolution of a pigment in water** <br>
  *Technologies:* Python, Keras, NumPy, Scikit-learn, Pandas, Plotly, Altair. <br>
  *Topics:* Fully connected network, PCA, time-series prediction. <br>

---

### <a href="https://github.com/bstollnitz/dieline-classifier">Package dieline classification</a>

<a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdlejxLv20qqllOVxA?e=jZh3nB">![](https://github.com/bstollnitz/dieline-classifier/blob/master/docs/poster.png?raw=true)</a>

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

--- 

### <a href="https://github.com/bstollnitz/pigment-in-water">Predicting time evolution of a pigment in water</a>

![](https://github.com/bstollnitz/pigment-in-water/blob/master/docs/pigment-in-water.png?raw=true)

The goal of this project is to explore techniques for predicting the behavior of food coloring in water. I started by recording several videos of the diffusion of pigment of colored candy immersed in water, to be used as data source. These videos can be found <a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEncETEzJ0kakis6g?e=p3rMWV">here</a>. 

I used PCA to reduce the dimensionality of the data, and then trained a feed-forward neural network based on several videos from the dataset. Once the network was trained, I used it to predict the behavior in an entire video starting from just the first frame. Given the first frame, the neural network predicts the second frame, which I feed back into the network to predict the third frame, and so on. You can see side-by-side comparisons of the original and predicted videos <a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEl_af5l_21j08xQA?e=JgxWFc">here</a>.

Next, I used a model discovery technique to find a partial differential equation (PDE) that describes the evolution of this physical phenomenon. The left-hand side of the PDE  is _u<sub>t</sub>_, and for the right-hand side, I considered a library of possible terms, such as _u<sub>x</sub>_, _u<sub>yy</sub>_, and _x u<sub>x</sub>_. I then used Lasso linear regression to find the appropriate coefficients for the terms, and obtained an equation that models the spreading of the food coloring. The main challenge of this task was the calculation of the derivative terms. I tried several techniques based on finite differences, but those led to noisy data and poor results. Therefore, I  decided to find derivatives by fitting a polynomial to the data within a small neighborhood around each pixel, and calculate the derivatives of the polynomial at the pixel of interest. Although this approach is more costly to compute, it produces much smoother results and a more accurate prediction.

If you'd like more details, you can read the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhaFKN0wKLCZaA6Rlug?e=zqRWE4">report</a> detailing my findings.

This was my final project for the Inferring Structure of Complex Systems class 
(AMATH 563) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.
