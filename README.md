# ChitraPy [![PyPI-Status](https://img.shields.io/badge/pypi-ChitraPy-blue)](https://pypi.org/project/ChitraPy) [![Build Status](https://travis-ci.com/mgautam98/ChitraPy.svg?branch=master)](https://travis-ci.com/mgautam98/ChitraPy)  [![codecov](https://codecov.io/gh/mgautam98/ChitraPy/branch/master/graph/badge.svg)](https://codecov.io/gh/mgautam98/ChitraPy) [![Documentation Status](https://readthedocs.org/projects/chitrapy/badge/?version=latest)](https://chitrapy.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/chitrapy)](https://pepy.tech/project/chitrapy)

 
ChitraPy is a digital Image Processing Library in Python.

# Installation
Install ChitraPy with:
```
pip install ChitraPy
```
Find the latest version on PyPi  
https://pypi.org/project/ChitraPy

### Install Dev version
```
git clone https://github.com/mgautam98/ChitraPy.git
cd ChitraPy
python3 setup.py install
```
## Documentation
https://chitrapy.readthedocs.io/en/latest/


## Usage
```
from ChitraPy import filters, helpers
import matplotlib.pyplot as plt

# Load a sample Image

!wget https://i.imgur.com/D24n5DL.png
img = plt.imread('./D24n5DL.png')
plt.imshow(img)

# invert an image
invert = filters.invert(img)
plt.imshow(invert)
```

## Features

* Crop
* Grayscale
* Negative
* Sepia
* Hue
* Salt and Pepper
* Stretch
* Warp
* Rotate
* Invert
* Gaussian blur
* Quick Blur
* Contrast Enhancement (Histogram Equalization)
* Local Contrast Enhancement
* Add Watermark
* Sobel Edge detection
* Halftoning
* Dither
* resize
* sharpen
* Emboss
* Identity
* Outline
* Monochrome
* Zoom
* Contrast adjustment (Histogram sliding)


## References

* https://en.wikipedia.org/wiki/Digital_image_processing
* http://setosa.io/ev/image-kernels/
* http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
* https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/dither/dither.pdf
* https://hypjudy.github.io/2017/03/19/dip-histogram-equalization/
