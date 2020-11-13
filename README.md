#### Overview

The following set of notebooks are meant to accompany the paper:
> Bowden, D. C., Sager, K., Fichtner A., Chmiel M. (2020). “Connecting Beamforming and Kernel-based Source Inversion”, Geophysical Journal International

These are *not* intended to reproduce every figure from the paper, but rather to be an illustrative example of some of the different methods: beamforming, Matched Field Processing, and a Noise Source Inversion (only the first iteration).

The notebooks are in two different directories, each specifying a different array:

./Bigger_array/  -> modelled after a subset of the Parkfield Array in California. This array of 11 stations is ideal for showing basic beamforming ("1_basic_beamforming.ipynb") and how precomputed noise correlations can be used for beamforming ("2_first_correlate_then_beamform.ipynb")

./Triangle_array/ -> a smaller array of only 3 stations. This smaller number makes it easier to visualize sensitivity kernels in Matched Field Processing ("3_mfp.ipynb") and a gradient-based noise source inversion ("4_kernel_based_inversion.ipynb").

Both directories include precomputed synthetics in ./array/OUTPUT/. These are generated with the code from Generalized Interferometery 1: https://cos.ethz.ch/software/research/gi.html. The code is included here, and notebooks "0_GI_intro_synthetics.ipynb" show how to get started, if you want to compute different synthetics by modifying an array, changing any correlation preprocessing, etc. Some of the more advanced topics in the paper, such as adjoint wavefields and a direct simulation of the noise correlation wavefield, are a part of that package (i.e., they are employed in "4_kernel_based_inversion.ipynb", but not demonstrated explicitely here).

The version of the GI code included here varies slightly from the main package. Mainly, the source distributions hardcoded into "source.py" are unique to the examples here, and so a new version of "source.py" is included in the directory "GI1_additions/". Additionally, a couple other functions are added to compute all station-station pairs of correlations in a more efficient manner, and to handle time-domain shifts of waveforms.

#### Installation

No explicit installation is required, assuming you can run a Jupyter Python Notebook, with python version 3.7, matplotlib, numpy and scipy.

To get started with Jupyter notebooks, we recommend using the conda or miniconda package manager: https://docs.conda.io/en/latest/miniconda.html

We also generally recommend using different environments, separate from the base or "root" environment. To install a new environment:

```
$ conda create -n beamforming_environment python=3.7 jupyter matplotlib numpy scipy
```

In this way, other packages can be added to your workflow (e.g., obspy, asdf, pandas, cartopy), without disrupting other projects or other environments. To activate the environment and start a new Jupyter Notebook Session in your webbrowser:

```
$ conda activate beamforming_environment
$ jupyter notebook
```

Last updated: November 2020. For questions or suggestions: daniel.bowden@erdw.ethz.ch

```python

```
