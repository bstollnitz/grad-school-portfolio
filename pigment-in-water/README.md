# Predicting time evolution of a pigment in water

*Technologies:* Python, Keras, NumPy, Scikit-learn, Pandas, Plotly, Altair. <br>
*Topics:* deep learning, principal component analysis (PCA), time-series, model discovery. <br>

## Description

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

## Running

To run this project on macOS:

```sh
conda env create -f environment_mac.yml
conda activate pigment-in-water
python main.py
```

To run this project on Windows:

```sh
conda env create -f environment_win.yml
conda activate pigment-in-water
python main.py
```
