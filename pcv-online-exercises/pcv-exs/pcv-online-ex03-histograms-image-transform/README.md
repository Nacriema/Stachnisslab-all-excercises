# Image Histograms

## 1. What is Image Histograms ? 
* The histogram show often the pixel intensity values occur in the image. 
* Histogram is a 2D bar plot where x-axis are the elements, y-axis is the occurrences for these items.
* With the RGB image, this tell how the distribution of intensity value along 3 color channels.

## 2. Usage of Image Histogram
* Analyse the image 
* Enhance the image
* Change the image
* Map intensity values into new intensity values


## 3. Manipulate Histogram
* Use a function that change the intensities of pixel -> change the histogram. These so called the point operators, they can
change the contrast, brightness and other properties of image.
* "Tone Curve": we can design tone curves so that the result image will get specific properties.
* Noise variance equalization: all pixels in the end have the same noise (this is used to analyse the image w.r.t the intensity value). 
* Histogram equalization: transform so that all bins of the target image histogram should have the same count -> Lead to 
a more contrast image. 

## 4. Where it is ? 
* Use in photography, photogrammetry, computer vision. 
* Use it as the pre-processing step for further algorithms. 

## 5. Lecture Note

### a. Image as a Function

For a Gray scale image: g(i, j): B -> G

with B: N -> N (B in range [0 ... I - 1, 0 ... J - 1]) and G = N (in range [0, 255])

### b. Histogram function
* Number of bins
* Def: h(g) = #number pixel with value g in the image. 
* Histogram function -> probability function: p(g) = h(g) / N (N is number of pixels in the image)

### c. Histogram Computation
* Iterating through the image, complexity is O(N) with N is number of pixels.
* Numpy already have this.

### d. Cumulative Histogram
* H(g) = Sum h(x) where x in range (0, g) 
* This is the integral of the histogram function
* Relation to PDF (Probability distribution function) and CDF (Cumulative distribution function) of the possible intensity 
values that occur in the image. (p(g) = h(g) / N as said above vs F(g) = H(g)/N )

### e. Histogram vs Information
* Tell us about the intensity overall of the image, (ex. change histogram -> change contrast, brightness (the mean and variance))
* Mean describe brightness, variance describe contrast, median is robust description of brightness. 

### f. Type of functions to change histogram
* Global operator
* Local operator
* Point operator


### g. Point Operator

b(i, j) = f(a(i, j), p) with b new image, a original image, p parameter 

#### Linear function form: 
b(i, j) = k + m * a(i, j)

k - shift the histogram - brightness

m - affect the contrast of the 
image (because it changes the range of pixel value).

The mean and deviation have changed: 
new_mean = k + m * old_mean  and new_deviation = |m| * old_deviation

This linear function form can be express in the "Tone Curve" as a straight line. Where x-axis is intensity of original, 
y-axis is value after applying the transform (so k is the intersection point of tone curve line and y-axis, m denote the
slope of line). Notice: m = -1 this is equal to flip the histogram (255 become 0 and so on...)

**Characteristic of Linear function**: When apply linear function to original image, we will lose some information, 
because all the pixel result which larger or smaller than threshold (here is 0 and 255) must be clipped !!!

#### Nonlinear functions:

* Thresholding: return black-white image (most case we do this). Then the tone curve has step function shape.
* Quantization: this is the general term of threshold, by given more return value base on the range of intensity in original image.
  (this is happens inside the chip of the camera to quantize the photon to intensity pixel)

#### Realizing Point Operators:
* This is efficiently computed using look-up table (1-D array where index is the input intensity and value at that index 
become the output intensity)
* 1-D array have fixed length, function evaluation corresponds to reading a byte from memory

### h. Color Images
* We build a histogram for each channel (3 channels have 3 histograms).
* Manipulate the individual channel with those function

