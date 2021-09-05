# Gabor transforms

*Technologies:* Python, NumPy, Plotly. <br>
*Topics:* Gabor transform, Fourier transform, spectrogram, signal processing, time-series. <br>

## Description

<p float="left">
  <img src="https://github.com/bstollnitz/gabor-transforms/blob/master/readme_files/mary_spectrograms.png?raw=true" width="400" />
  <img src="https://github.com/bstollnitz/gabor-transforms/blob/master/readme_files/zoomed_filtered_spectrograms.png?raw=true" width="400" />
</p>

I analyze three audio files in this project. The first file contains music by Handel, and the two other contain the song ``Mary had a little lamb'' played on the piano and recorder. 

For the first audio file, my goal is to compare the effects of the application of different Gabor transforms. I produce several spectrograms by using a wide range of Gabor filters with different shapes, widths, and time steps. I analyze the resulting spectrograms and point out the compromises involved in making different choices.

For the second and third audio files, my goal is to produce music scores for the tune. I start by visualizing the logarithms of the spectrograms, then I simplify the data, remove overtones, and select a better frequency range for the visualization. The result is an image representation of the frequencies of each note played in the tune.

I conclude that Gabor transforms are effective at analyzing time-series data where both the frequency and time information are important.

You can find more details in the <a href="https://1drv.ms/b/s!AiCY1Uw6PbEfhd0SyGM9M1QW6TqZmA?e=ipmpvq">report</a> for this project.

This was a homework assignment for the Computational Methods for Data Analysis class (AMATH 582) at the University of Washington, which I completed as part 
of my masters in Applied Mathematics.

## Running

To run this project on macOS or Windows:

```sh
conda env create -f environment.yml
conda activate gabor-transforms
python main.py
```
