# Bea Stollnitz projects

This public repository contains machine learning and data science related projects completed by Bea Stollnitz. All code is done in Python.

You can find below a description of each project.

# <h3><a href="https://github.com/bstollnitz/portfolio/tree/master/PigmentInWater">Predicting time evolution of a pigment in water</a></h3>

The goal for this project is to explore techniques for predicting the behavior of food coloring in water. I started by recording 
several videos of the diffusion of pigment of colored candy immersed in water, to be used as data source. These videos can be found 
<a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEncETEzJ0kakis6g?e=p3rMWV">here</a>. 

I used PCA to reduce the dimensionality of the data in an optimal way, and then trained a feed forward neural network based
on several videos from the dataset. After training, given a frame in a video, the neural network was
able to predict the next frame. I then gave it as input the initial frame of a test video to predict the second, and 
gave it subsequent predictions to obtain an entire video. 
Side-by-side actual and predicted videos showing the results obtained can be found 
<a href="https://1drv.ms/u/s!AiCY1Uw6PbEfhaEl_af5l_21j08xQA?e=JgxWFc">here</a>.

I then used a model discovery technique to find a partial differential equation (PDE) that describes the evolution of this physical
phenomenon. The left-hand side of the PDE considered is u<sub>t</sub>, and for the right-hand side, I 
considered a library of possible terms, such as u<sub>x</sub>, u<sub>yy</sub> and x * u<sub>x</sub>. I then used lasso 
linear regression to find the appropriate coefficients for the terms, and with that, obtain an equation that models the spreading of the
food coloring. The main challenge of this task was the calculation of the derivative terms. I tried several techniques based on 
finite-differences, but those lead to noisy data and poor results. I then decided to find derivatives by calculating a polynomial 
for each pixel based on a cube of data around the pixel, and determining its derivatives at the pixel of interest. Although more costly 
to compute, this technique produced much smoother results and a more accurate prediction.

A complete report detailing my findings can be found <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhaFKN0wKLCZaA6Rlug?e=zqRWE4">here</a>.

This was my final project for the AMATH 563 class at the University of Washington, which I completed as part of my masters in 
Applied Mathematics.
