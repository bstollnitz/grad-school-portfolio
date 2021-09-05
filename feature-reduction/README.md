# Feature reduction

*Technologies:* Python, NumPy, Plotly, OpenCV, CSRT object tracker. <br>
*Topics:* principal component analysis (PCA), dimensionality reduction, object tracking, 
time-series. <br>

## Description

<p float="left">
  <img src="https://github.com/bstollnitz/feature-reduction/blob/master/readme_files/3_dominant_modes.png?raw=true" width="400" />
</p>

In this project, I analyze twelve videos in which three cameras recorded four different scenes: an object oscillating with vertical displacement only, a similar scene but with significant camera shake, an object oscillating with horizontal and vertical displacement, and an object that rotates in addition to oscillating horizontally and vertically. I use OpenCV's CSRT object tracker to follow the object and obtain trajectories for each camera. I then combine the trajectory data of different cameras for each scenario by including x and y feature information as the rows of our data matrix, with columns corresponding to the temporal dimension. I perform Principal Component Analysis (PCA) using Singular Value Decomposition (SVD), and analyze the potential for dimensionality reduction of the data for each scenario.

I observe that scenarios 1 and 3 present little ambiguity. Scenario 1 clearly needs only a single spatial mode to describe its behavior, which matches my intuition because the object only moves vertically (and therefore a single coordinate is enough to completely describe its movement). Scenario 3 needs two spatial modes, which also matches my intuition because the object moves both horizontally and vertically, roughly in a plane. Projecting the original data into the corresponding number of spatial modes significantly reduces the size of the data by removing redundancy, while keeping the more expressive information intact.

Scenarios 2 and 4 are less intuitive, but the computation results are no less interesting. Scenario 2 can be represented using either one or three spatial modes. One mode would represent its vertical displacement, but the camera shake adds a new level of complexity to the movement that may need three spatial modes to encode. In Scenario 4, the object motion is very pronounced in the vertical direction, and less pronounced in the horizontal direction, quickly decaying to just vertical motion. As a result, we could choose either one or two modes to represent the motion.

I conclude that PCA is an effective method for dimensionality reduction of data. This method enables us to encode the data in a fraction of the space by removing redundancy while keeping all the relevant information.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd1bqRoxfimQY2tlnQ?e=1iYMTK">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project on macOS or Windows:

```sh
conda env create -f environment.yml
conda activate feature-reduction
python main.py
```
