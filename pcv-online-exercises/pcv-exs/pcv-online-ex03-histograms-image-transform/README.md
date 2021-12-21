# Image Histograms

##1. What is Image Histograms ? 
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