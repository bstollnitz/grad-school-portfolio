# Bea Stollnitz projects

Here's a list of my machine learning, signal processing, and data science projects, with details below.

- **Package dieline classification** <br>
*Technologies:* Python, Keras, NumPy, Scikit-learn, Plotly. <br>
*Topics:* deep learning, convolutional neural network (CNN), classification. <br>

- **Human activity classification** <br>
*Technologies:* Python, PyTorch, PyWavelets, NumPy, Plotly, Tensorboard, H5py, Tqdm, Pillow. <br>
*Topics:* deep learning, convolutional neural network (CNN), continuous wavelet transform, Gabor transform, classification, signal processing, time-series. <br>

- **Predicting time evolution of a pigment in water** <br>
*Technologies:* Python, Keras, NumPy, Scikit-learn, Pandas, Plotly, Altair. <br>
*Topics:* deep learning, principal component analysis (PCA), time-series, model discovery. <br>

- **Music classification using LDA** <br>
*Technologies:* Python, NumPy, Plotly, Scikit-learn, Scipy. <br>
*Topics:* linear discriminant analysis (LDA), classification, principal 
component analysis (PCA), Gabor transform, time-series. <br>

- **Eigenfaces** <br>
*Technologies:* Python, NumPy, Plotly, Scikit-learn, Pillow. <br>
*Topics:* principal component analysis (PCA), support vector machine (SVM),
classification. <br>

- **Gabor transforms** <br>
*Technologies:* Python, NumPy, Plotly. <br>
*Topics:* Gabor transform, Fourier transform, spectrogram, signal processing, time-series. <br>

- **Feature reduction** <br>
*Technologies:* Python, NumPy, Plotly, OpenCV, CSRT object tracker. <br>
*Topics:* principal component analysis (PCA), dimensionality reduction, object tracking, 
time-series. <br>

- **Separation of background and foreground using DMD** <br>
*Technologies:* Python, NumPy, Plotly, OpenCV. <br>
*Topics:* dynamic mode decomposition (DMD), video processing. <br>

- **Denoising 3D scanned data** <br>
*Technologies:* Python, NumPy, Plotly. <br>
*Topics:* fast Fourier transform (FFT), time-series. <br>

---

### Package dieline classification

<a href="/dieline-classifier">GitHub folder with code</a>

<a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhdlejxLv20qqllOVxA?e=jZh3nB">![](/dieline-classifier/docs/poster.png?raw=true)</a>

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

### Human activity classification

<a href="/activity-classification">GitHub folder with code</a>

<p float="left">
  <img src="/activity-classification/readme_files/spectrograms.png" width="700" />
</p>

In this project, I use three different approaches to classify temporal signals according to the associated activity. The input data consists of several thousand short snippets of measurements obtained from nine sensors (such as acceleration and gyroscope) while people performed six different activities (such as walking or sitting). In my first approach, I train a simple feed-forward network using the raw temporal signals and associated labels. In my second approach, I compute spectrograms by applying a Gabor transform to the temporal signals, and train a CNN to classify the spectrograms. In my third approach, I compute scaleograms by using a continuous wavelet transform, and train a CNN to classify the scaleograms.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfheUz1u-obevm2AsltA?e=U0Ls2N">report</a> for this project.

This was my final project for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

--- 

### Predicting time evolution of a pigment in water

<a href="/pigment-in-water">GitHub folder with code</a>

<p float="left">
    <img src="/pigment-in-water/readme_files/pigment-in-water.png" width="800"/>
</p>

The goal of this project is to explore techniques for predicting the behavior of food coloring in water. I started by recording several videos of the diffusion of pigment of colored candy immersed in water, to be used as data source. 

I used PCA to reduce the dimensionality of the data, and then trained a feed-forward neural network based on several videos from the dataset. Once the network was trained, I used it to predict the behavior in an entire video starting from just the first frame. Given the first frame, the neural network predicts the second frame, which I feed back into the network to predict the third frame, and so on.

Next, I used a model discovery technique to find a partial differential equation (PDE) that describes the evolution of this physical phenomenon. The left-hand side of the PDE  is _u<sub>t</sub>_, and for the right-hand side, I considered a library of possible terms, such as _u<sub>x</sub>_, _u<sub>yy</sub>_, and _x u<sub>x</sub>_. I then used Lasso linear regression to find the appropriate coefficients for the terms, and obtained an equation that models the spreading of the food coloring. The main challenge of this task was the calculation of the derivative terms. I tried several techniques based on finite differences, but those led to noisy data and poor results. Therefore, I  decided to find derivatives by fitting a polynomial to the data within a small neighborhood around each pixel, and calculate the derivatives of the polynomial at the pixel of interest. Although this approach is more costly to compute, it produces much smoother results and a more accurate prediction.

If you'd like more details, you can read the <a href='https://1drv.ms/b/s!AiCY1Uw6PbEfheU025caSHVd7gzJYA?e=Uiw5pZ'>report</a> detailing my findings.

This was my final project for the Inferring Structure of Complex Systems class 
(AMATH 563) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---

### Music classification using LDA

<a href="/music-classification">GitHub repo with code</a>

<p float="left">
  <img src="/music-classification//readme_files/reduced_subspace_bands_of_classical_genre.png?raw=true" width="600" />
</p>

In this project, I classify 108 music clips from four different genres and three different bands per genre, according to band name and genre. I first create a spectrogram per music clip using a Gabor transform with a Gaussian filter. I then reduce the dimensionality of the spectrograms using SVD and PCA. And last, I classify each clip using my own custom implementation of linear discriminant analysis (LDA). I also classify them using scikit-learn's LDA classification to ensure I'm on the right track.

In my LDA implementation, I solve a generalized eigenvalue problem with the between-class covariance matrix and the within-class covariance matrix. The resulting eigenvectors provide a transformation that maximizes the separation between class centroids while minimizing the variance within classes. I then classify each test point by applying this transformation and finding the closest class centroid. I show how to visualize the process in a 2D plot, using a Voronoi diagram to illustrate the classification boundaries.

From my experiments I conclude that LDA is quite effective at classifying audio clips by band name across different genres, and by band name even within a single genre. However, for the data set that I used, LDA struggled to effectively classify audio clips by genre. The results I obtained from my LDA implementation are consistent with results from the scikit-learn implementation.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd5GXNTlLOwEI62cvA?e=ej1ho2">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---


### Eigenfaces

<a href="/eigenfaces">GitHub repo with code</a>

<p float="left">
  <img src="/eigenfaces/readme_files/eigenfaces.png" width="800" />
</p>

In this project, I consider two data sets, one containing images of faces that have been cropped and aligned, and another containing images of faces that have not been aligned. I decompose this data using principal component analysis (PCA), and analyze the energy, coefficients, and visual representation of the data's spatial modes. I notice how the most relevant modes seem to capture changes in the lighting and position of the faces above all.

I show how PCA can be used in two applications: image compression and image classification. For the compression scenario, I choose an image from each dataset, and reconstruct it using increasing numbers of modes. The more modes I use, the better the approximation is to the original image. I compare the mean squared error between the original image and an image reconstructed with 50 modes, for each dataset, and conclude that the error is smaller for the cropped images. The cropped images have the face's features well aligned, so it's natural that a better representation would be achieved with the same number of modes.

For the image classification scenario, I split the data into training and test sets, use PCA to find the 50 most informative modes for the training data, and use those modes as a basis to reduce both training and test data. I then use the support vector machine (SVM) method to classify which subject is photographed in each of the test images. I achieve very high accuracies for both the cropped and uncropped images. I conclude that just a few modes are sufficient to capture the identity of each photo, at least for these small data sets.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd57di7RA0VqFnOq4Q?e=oEApsp">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.


---

### Gabor transforms

<a href="/gabor-transforms">GitHub repo with code</a>

<p float="left">
  <img src="/gabor-transforms/readme_files/mary_spectrograms.png" width="400" />
  <img src="/gabor-transforms/readme_files/zoomed_filtered_spectrograms.png" width="400" />
</p>

I analyze three audio files in this project. The first file contains music by Handel, and the two other contain the song ``Mary had a little lamb'' played on the piano and recorder. 

For the first audio file, my goal is to compare the effects of the application of different Gabor transforms. I produce several spectrograms by using a wide range of Gabor filters with different shapes, widths, and time steps. I analyze the resulting spectrograms and point out the compromises involved in making different choices.

For the second and third audio files, my goal is to produce music scores for the tune. I start by visualizing the logarithms of the spectrograms, then I simplify the data, remove overtones, and select a better frequency range for the visualization. The result is an image representation of the frequencies of each note played in the tune.

I conclude that Gabor transforms are effective at analyzing time-series data where both the frequency and time information are important.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd0SyGM9M1QW6TqZmA?e=ipmpvq">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---

### Feature reduction

<a href="/feature-reduction">GitHub repo with code</a>

<p float="left">
  <img src="/feature-reduction/readme_files/3_dominant_modes.png" width="400" />
</p>

In this project, I analyze twelve videos in which three cameras recorded four different scenes: an object oscillating with vertical displacement only, a similar scene but with significant camera shake, an object oscillating with horizontal and vertical displacement, and an object that rotates in addition to oscillating horizontally and vertically. I use OpenCV's CSRT object tracker to follow the object and obtain trajectories for each camera. I then combine the trajectory data of different cameras for each scenario by including x and y feature information as the rows of our data matrix, with columns corresponding to the temporal dimension. I perform Principal Component Analysis (PCA) using Singular Value Decomposition (SVD), and analyze the potential for dimensionality reduction of the data for each scenario.

I observe that scenarios 1 and 3 present little ambiguity. Scenario 1 clearly needs only a single spatial mode to describe its behavior, which matches my intuition because the object only moves vertically (and therefore a single coordinate is enough to completely describe its movement). Scenario 3 needs two spatial modes, which also matches my intuition because the object moves both horizontally and vertically, roughly in a plane. Projecting the original data into the corresponding number of spatial modes significantly reduces the size of the data by removing redundancy, while keeping the more expressive information intact.

Scenarios 2 and 4 are less intuitive, but the computation results are no less interesting. Scenario 2 can be represented using either one or three spatial modes. One mode would represent its vertical displacement, but the camera shake adds a new level of complexity to the movement that may need three spatial modes to encode. In Scenario 4, the object motion is very pronounced in the vertical direction, and less pronounced in the horizontal direction, quickly decaying to just vertical motion. As a result, we could choose either one or two modes to represent the motion.

I conclude that PCA is an effective method for dimensionality reduction of data. This method enables us to encode the data in a fraction of the space by removing redundancy while keeping all the relevant information.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd1bqRoxfimQY2tlnQ?e=1iYMTK">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---

### Separation of background and foreground using DMD

<a href="/dmd-separation">GitHub repo with code</a>

<p float="left">
  <a href="https://bea-portfolio.s3-us-west-2.amazonaws.com/dmd-separation/monkey-giraffe.mp4">
    <img src="/dmd-separation/readme_files/video.png" width="800" controls/>
  </a>
</p>

<p float="left">
  <img src="/dmd-separation/readme_files/monkey-giraffe_2_background.png" width="400" />
  <img src="/dmd-separation/readme_files/monkey-giraffe_2_foreground.png" width="400" />
</p>

In this project, I explore two techniques for separating a moving foreground from a stationary background in videos. Both techniques rely on Dynamic Mode Decomposition (DMD), following somewhat different steps to arrive at the foreground and background pixel values. I use three videos of animal puppets exploring the Seattle cityscape in this analysis. No animals were harmed in this research.

In the first technique, I follow the approach described by <a href="https://arxiv.org/abs/1404.7592">Grosek and Kutz</a>. I calculate a stationary background frame by keeping only the DMD mode with the eigenvalue of smallest magnitude. I then compute the foreground by subtracting the background mode from each frame of the original video. And finally, I extract all the negative values in the foreground into a residual matrix R, which I subtract from the foreground and add to the background. This technique has the convenient property that the background and foreground add up to the original video, but produces poor results for our particular data set. This is likely because the moving foreground in our videos is frequently darker than the background.

In the second technique, I relax the requirement that the foreground and background must add up to the original video. I again compute the stationary background from the DMD mode with the eigenvalue of smallest magnitude, but without any adjustment from the residual matrix. The foreground pixels are assigned values of zero wherever the original video is similar to the background, and the original pixel values elsewhere. This technique produces better results for our data set, giving a cleanly separated background image and an imperfect but tolerable foreground video.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfheEEpuyj0ONiHQuIww?e=i1rv5K">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

---

### Denoising 3D scanned data

<a href="/denoising-3D-scans">GitHub repo with code</a>

<img src="/denoising-3D-scans/readme_files/marble-path.png" width="350"/>

In this project, I denoise a series of three-dimensional scanned data to determine the path of a marble ingested by a dog.
I accomplish this by converting the data into the frequency domain using an FFT, averaging the spectra over time, and using
the average spectrum to construct a Gaussian filter. I then denoise the data in the frequency domain using the Gaussian filter,
and convert it back into the spatial domain. The location of the marble can then be found by looking for the peak density in the
spatial data.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd0M6ubdeAjrBPuUJw?e=ejklwQ">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.