###This writeup is based on the writeup template from the [project repo](https://github.com/udacity/CarND-Vehicle-Detection/blob/master).

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.JPG
[image2]: ./output_images/HOG_example.JPG
[image3]: ./output_images/sliding_windows.JPG
[video1]: ./result_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Feature extraction based on HOG.

The code for this step is contained in the 2nd and 3rd code cells of the IPython notebook.  

At first all the `vehicle` and `non-vehicle` images have been loaded in arrays.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) has been applied on images from each of the two classes.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Number of car images:  8792
Number of not car images:  8968

The functions from the lesson has been adapted in this project.

####2. Final set of HOG parameters.

The following combination of parameters has been used to train classifier.

The HLS color space has been used in order to extract the features for classification. The hue, saturation and lightness discribe intuitivelly different aspects of the color, and therefore help to separate the different features.  So the RGB to HLS convertion has been applied to the input image.

Parameters:
- all three channels of the HLS color space
- number of orientation bins: 9 (it is easiers to divide 360 degree with 9)
- block of cells: (2x2)
- cell sizes: (8x8)
- the "gamma" normalization schema has been used by the block normalization in order to reduce the effects of shadows or other illumination variation.

####3. Traininig of a classifier

The code for this step is contained in the 5th and 6th code cells of the IPython notebook. 

The training of the linear SVM using Spatial, Histogram and HOG features. 

Spatial binning dimensions:
`spatial_size` = (32, 32)
Number of histogram bins
`hist_bins` = 32

Because the combination of the different features has been used, the normalization is required in order to avoid the domination of the features with higher amplitude. To normalize the feature vectors for training `sklearn.preprocessing.StandardScaler()` has been utilized. 

The length of the feature vector is: __8460__ 
For random shuffeling of the data `sklearn.model_selection.train_test_split` has been used, with the train/test ratio 80:20.  

The test accuracy of __98.93%__ has been achieved for Linear SVC.

To determine the best performing parameter set for the SVC-classifier the exhaustive search over specified parameter values for an estimator `sklearn.model_selection.GridSearchCV` with following parameter grid has been applied:
`
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
`
To save the time only the part of the data set has been used here (1000 randomly defined samples of car and not-car). The result: __{'kernel': 'rbf', 'C': 1, 'gamma': 0.0001}__ 

So the classifier has been changed and the test accuracy of __99.66%__ has been derived. 

NOTE: But as it has been recognized afterwards - such classifier is very slow in processing and linear classifier has been used in the last implementation.  

###Sliding Window Search

####1. Sliding window search.  

The code for this step is contained in the 4th and 8th code cells of the IPython notebook. The functions from the lesson have been applied. 

To reduce the search area the min and max values for y-coordinate has been defined: 
`y_start_stop = [400, 670]`

After some tuning the best overlap and window size has been choosen as follows:
`overlap = 0.75`
`search_win = 96`

![alt text][image3]

As it can be seen in figures above the vehicles have been detected entirely. That was the criteria for parameter choosing.  

####2. Pipeline.  

The code for this step is contained in 11th (for test) and in 16th code cells of the IPython notebook. 

The HOG-sub sampling with two scales = 1.7 and 1.2 using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector has been used to avoid multiple HOG calculations. The function `find_cars` from the lesson has been applied to extract features using hog sub-sampling and make predictions.

The filtering of false positives have been produced (s. details below).

---

### Video Implementation

####1. Final video output.  

All objects in the project video are detected without drop outs and most false positives are filtered out.

Here's a [link to my video result](./result_project_video.mp4)


####2. Filter for false positives and multiple detections

In order to optimize the performance of the classifier and remove false positives, a heat-map from the positive detections of the consecutive frames has been created and then thresholded. The thresholding has been performed twice:
 - 1st threshold = 1 overlappings to help remove false positives by means windows in one frame
 - 2nd threshold = 6 overlappings to remove false positive over 10 consecutive frames

To associate the individual blobs in the heatmap the `scipy.ndimage.measurements.label()` has been used. This helps to combine overlapping detections.

---

###Discussion

####1. Problems / issues in implementation of this project.  

The current implementation will probably fail: 
- on roads with higher slope the detection range will be reduced
- by vehicles like truck, motorcycle (because such object types have not been used by training of the classifier)
- pedestrians will not be detected (the same reason as before)
- objects with higher difference in relative speed to the ego-vehicle could produce the multiple objects due to restictions of the labeling-algorithm.

To make it more robust:
- the data set can be increased with more samples of the specific objects 
- sophisticated tracking algorithm (like kalman-filter) can be applied which filters some parameters over time.