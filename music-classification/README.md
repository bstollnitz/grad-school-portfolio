# Music classification using LDA

*Technologies:* Python, NumPy, Plotly, Scikit-learn, Scipy. <br>
*Topics:* linear discriminant analysis (LDA), classification, principal 
component analysis (PCA), Gabor transform, time-series. <br>

## Description

<p float="left">
  <img src="https://github.com/bstollnitz/music-classification/blob/master/readme_files/reduced_subspace_bands_of_classical_genre.png?raw=true" width="600" />
</p>

In this project, I classify 108 music clips from four different genres and three different bands per genre, according to band name and genre. I first create a spectrogram per music clip using a Gabor transform with a Gaussian filter. I then reduce the dimensionality of the spectrograms using SVD and PCA. And last, I classify each clip using my own custom implementation of linear discriminant analysis (LDA). I also classify them using scikit-learn's LDA classification to ensure I'm on the right track.

In my LDA implementation, I solve a generalized eigenvalue problem with the between-class covariance matrix and the within-class covariance matrix. The resulting eigenvectors provide a transformation that maximizes the separation between class centroids while minimizing the variance within classes. I then classify each test point by applying this transformation and finding the closest class centroid. I show how to visualize the process in a 2D plot, using a Voronoi diagram to illustrate the classification boundaries.

From my experiments I conclude that LDA is quite effective at classifying audio clips by band name across different genres, and by band name even within a single genre. However, for the data set that I used, LDA struggled to effectively classify audio clips by genre. The results I obtained from my LDA implementation are consistent with results from the scikit-learn implementation.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd5GXNTlLOwEI62cvA?e=ej1ho2">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project on macOS or Windows:

```sh
conda env create -f environment.yml
conda activate music-classification
python main.py
```
