# Eigenfaces

*Technologies:* Python, NumPy, Plotly, Scikit-learn, Pillow. <br>
*Topics:* principal component analysis (PCA), support vector machine (SVM),
classification. <br>

## Description

<p float="left">
  <img src="https://github.com/bstollnitz/eigenfaces/blob/master/readme_files/eigenfaces.png?raw=true" width="800" />
</p>

In this project, I consider two data sets, one containing images of faces that have been cropped and aligned, and another containing images of faces that have not been aligned. I decompose this data using principal component analysis (PCA), and analyze the energy, coefficients, and visual representation of the data's spatial modes. I notice how the most relevant modes seem to capture changes in the lighting and position of the faces above all.

I show how PCA can be used in two applications: image compression and image classification. For the compression scenario, I choose an image from each dataset, and reconstruct it using increasing numbers of modes. The more modes I use, the better the approximation is to the original image. I compare the mean squared error between the original image and an image reconstructed with 50 modes, for each dataset, and conclude that the error is smaller for the cropped images. The cropped images have the face's features well aligned, so it's natural that a better representation would be achieved with the same number of modes.

For the image classification scenario, I split the data into training and test sets, use PCA to find the 50 most informative modes for the training data, and use those modes as a basis to reduce both training and test data. I then use the support vector machine (SVM) method to classify which subject is photographed in each of the test images. I achieve very high accuracies for both the cropped and uncropped images. I conclude that just a few modes are sufficient to capture the identity of each photo, at least for these small data sets.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd57di7RA0VqFnOq4Q?e=oEApsp">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project on macOS or Windows:

```sh
conda env create -f environment.yml
conda activate eigenfaces
python main.py
```
