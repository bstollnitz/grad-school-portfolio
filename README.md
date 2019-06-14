# Bea Stollnitz projects

This repository contains machine learning and data science related projects completed by Bea Stollnitz. All code is in Python.

### <a href="https://github.com/bstollnitz/portfolio/tree/master/PigmentInWater">Predicting time evolution of a pigment in water</a>

![](Images/PigmentInWater.PNG?raw=true)

The goal of this project is to explore techniques for predicting the behavior of food coloring in water. I started by recording several videos of the diffusion of pigment of colored candy immersed in water, to be used as data source. These videos can be found <a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEncETEzJ0kakis6g?e=p3rMWV">here</a>. 

I used PCA to reduce the dimensionality of the data, and then trained a feed-forward neural network based on several videos from the dataset. Once the network was trained, I used it to predict the behavior in an entire video starting from just the first frame. Given the first frame, the neural network predicts the second frame, which I feed back into the network to predict the third frame, and so on. You can see side-by-side comparisons of the original and predicted videos <a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEl_af5l_21j08xQA?e=JgxWFc">here</a>.

Next, I used a model discovery technique to find a partial differential equation (PDE) that describes the evolution of this physical phenomenon. The left-hand side of the PDE  is _u<sub>t</sub>_, and for the right-hand side, I considered a library of possible terms, such as _u<sub>x</sub>_, _u<sub>yy</sub>_, and _x u<sub>x</sub>_. I then used Lasso linear regression to find the appropriate coefficients for the terms, and obtained an equation that models the spreading of the food coloring. The main challenge of this task was the calculation of the derivative terms. I tried several techniques based on finite differences, but those led to noisy data and poor results. I  decided to find derivatives by fitting a polynomial to the data within a small neighborhood around each pixel, and calculate the derivatives of the polynomial at the pixel of interest. Although this approach is more costly to compute, it produces much smoother results and a more accurate prediction.

If you'd like more details, you can read the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhaFKN0wKLCZaA6Rlug?e=zqRWE4">report</a> detailing my findings.

This was my final project for the AMATH 563 class at the University of Washington, which I completed as part of my masters in Applied Mathematics.
