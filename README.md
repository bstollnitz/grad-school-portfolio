# Bea Stollnitz projects

Here's a list of my machine learning, deep learning, signal processing, and data science projects, with details below.

- **Package dieline classification** <br>
  *Technologies:* Python, Keras, NumPy, Scikit-learn, Plotly. <br>
  *Topics:* deep learning, convolutional neural network (CNN), classification. <br>

- **Predicting time evolution of a pigment in water** <br>
*Technologies:* Python, Keras, NumPy, Scikit-learn, Pandas, Plotly, Altair. <br>
*Topics:* deep learning, principal component analysis (PCA), time-series, model discovery. <br>

- **Eigenfaces** <br>
*Technologies:* Python, NumPy, Plotly, Scikit-learn, Pillow. <br>
*Topics:* principal component analysis (PCA), support vector machine (SVM),
classification. <br>

<!--
- **Denoising of 3D scanned data** <br>
  *Technologies:* Python, NumPy, Plotly. <br>
  *Topics:* time-series, FFT, denoising by filtering. <br>
-->
---

### Package dieline classification

<a href="https://github.com/bstollnitz/dieline-classifier">GitHub repo with code</a>

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

### Predicting time evolution of a pigment in water

<a href="https://github.com/bstollnitz/pigment-in-water">GitHub repo with code</a>

<p float="left">
    <img src="https://github.com/bstollnitz/pigment-in-water/blob/master/readme_files/pigment-in-water.png" width="800"/>
</p>

The goal of this project is to explore techniques for predicting the behavior of food coloring in water. I started by recording several videos of the diffusion of pigment of colored candy immersed in water, to be used as data source. 

I used PCA to reduce the dimensionality of the data, and then trained a feed-forward neural network based on several videos from the dataset. Once the network was trained, I used it to predict the behavior in an entire video starting from just the first frame. Given the first frame, the neural network predicts the second frame, which I feed back into the network to predict the third frame, and so on.

Next, I used a model discovery technique to find a partial differential equation (PDE) that describes the evolution of this physical phenomenon. The left-hand side of the PDE  is _u<sub>t</sub>_, and for the right-hand side, I considered a library of possible terms, such as _u<sub>x</sub>_, _u<sub>yy</sub>_, and _x u<sub>x</sub>_. I then used Lasso linear regression to find the appropriate coefficients for the terms, and obtained an equation that models the spreading of the food coloring. The main challenge of this task was the calculation of the derivative terms. I tried several techniques based on finite differences, but those led to noisy data and poor results. Therefore, I  decided to find derivatives by fitting a polynomial to the data within a small neighborhood around each pixel, and calculate the derivatives of the polynomial at the pixel of interest. Although this approach is more costly to compute, it produces much smoother results and a more accurate prediction.

If you'd like more details, you can read the <a href='https://1drv.ms/b/s!AiCY1Uw6PbEfheU025caSHVd7gzJYA?e=Uiw5pZ'>report</a> detailing my findings.

This was my final project for the Inferring Structure of Complex Systems class 
(AMATH 563) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---

### Eigenfaces

<a href="https://github.com/bstollnitz/eigenfaces">GitHub repo with code</a>

<p float="left">
  <img src="https://github.com/bstollnitz/eigenfaces/blob/master/readme_files/eigenfaces.png?raw=true" width="800" />
</p>

In this project, I consider two data sets, one containing images of faces that have been cropped and aligned, and another containing images of faces that have not been aligned. I decompose this data using principal component analysis (PCA), and analyze the energy, coefficients, and visual representation of the data's spatial modes. I notice how the most relevant modes seem to capture changes in the lighting and position of the faces above all.

I show how PCA can be used in two applications: image compression and image classification. For the compression scenario, I choose an image from each dataset, and reconstruct it using increasing numbers of modes. The more modes I use, the better the approximation is to the original image. I compare the mean squared error between the original image and an image reconstructed with 50 modes, for each dataset, and conclude that the error is smaller for the cropped images. The cropped images have the face's features well aligned, so it's natural that a better representation would be achieved with the same number of modes.

For the image classification scenario, I split the data into training and test sets, use PCA to find the 50 most informative modes for the training data, and use those modes as a basis to reduce both training and test data. I then use the support vector machine (SVM) method to classify which subject is photographed in each of the test images. I achieve very high accuracies for both the cropped and uncropped images. I conclude that just a few modes are sufficient to capture the identity of each photo, at least for these small data sets.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd57di7RA0VqFnOq4Q?e=oEApsp">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

<!-- ### Denoising 3D scanned data

<a href="https://github.com/bstollnitz/denoising-3D-scans">GitHub repo with code</a>

<img src="https://github.com/bstollnitz/denoising-3D-scans/blob/master/docs/marble-path.png?raw=true" width="350"/>

In this project, I denoise a series of three-dimensional scanned data to determine the path of a marble ingested by a dog.
I accomplish this by converting the data into the frequency domain using an FFT, averaging the spectra over time, and using
the average spectrum to construct a Gaussian filter. I then denoise the data in the frequency domain using the Gaussian filter,
and convert it back into the spatial domain. The location of the marble can then be found by looking for the peak density in the
spatial data.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdxWL0lSvCZrnEAEBA?e=46bby1">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics. -->
