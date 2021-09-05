# Separation of background and foreground using DMD

*Technologies:* Python, NumPy, Plotly, OpenCV. <br>
*Topics:* dynamic mode decomposition (DMD), video processing. <br>

## Description

<p float="left">
  <a href="https://bea-portfolio.s3-us-west-2.amazonaws.com/dmd-separation/monkey-giraffe.mp4">
    <img src="readme_files/video.png" width="800" controls/>
  </a>
</p>

<p float="left">
  <img src="https://github.com/bstollnitz/dmd-separation/blob/master/readme_files/monkey-giraffe_2_background.png?raw=true" width="400" />
  <img src="https://github.com/bstollnitz/dmd-separation/blob/master/readme_files/monkey-giraffe_2_foreground.png?raw=true" width="400" />
</p>

In this project, I explore two techniques for separating a moving foreground from a stationary background in videos. Both techniques rely on Dynamic Mode Decomposition (DMD), following somewhat different steps to arrive at the foreground and background pixel values. I use three videos of animal puppets exploring the Seattle cityscape in this analysis. No animals were harmed in this research.

In the first technique, I follow the approach described by <a href="https://arxiv.org/abs/1404.7592">Grosek and Kutz</a>. I calculate a stationary background frame by keeping only the DMD mode with the eigenvalue of smallest magnitude. I then compute the foreground by subtracting the background mode from each frame of the original video. And finally, I extract all the negative values in the foreground into a residual matrix R, which I subtract from the foreground and add to the background. This technique has the convenient property that the background and foreground add up to the original video, but produces poor results for our particular data set. This is likely because the moving foreground in our videos is frequently darker than the background.

In the second technique, I relax the requirement that the foreground and background must add up to the original video. I again compute the stationary background from the DMD mode with the eigenvalue of smallest magnitude, but without any adjustment from the residual matrix. The foreground pixels are assigned values of zero wherever the original video is similar to the background, and the original pixel values elsewhere. This technique produces better results for our data set, giving a cleanly separated background image and an imperfect but tolerable foreground video.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfheEEpuyj0ONiHQuIww?e=i1rv5K">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project on macOS or Windows:

```sh
conda env create -f environment.yml
conda activate dmd-separation
python main.py
```
