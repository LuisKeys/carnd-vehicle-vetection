import numpy as np
import glob
import detection_library as dl
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from random import shuffle
import time



# Divide into cars and not_cars
def get_data():

	# Main function internal parameters
	number_of_samples = 500
	path_non_cars = './non-cars_smallset/'
	path_cars = './cars_smallset/'

	not_cars_images = glob.glob(path_non_cars + '*.jpeg')
	cars_images = glob.glob(path_cars + '*.jpeg')
	x = [[i] for i in range(10)]
	shuffle(not_cars_images)
	shuffle(cars_images)

	cars = []
	not_cars = []
	counter = 0
	for image in not_cars_images:
	    if counter < number_of_samples:
	        not_cars.append(image)
	        counter += 1

	counter = 0
	for image in cars_images:
	    if counter < number_of_samples:
	        cars.append(image)
	        counter += 1

	return cars, not_cars

# Extract features from a list of images
def extract_features(imgs, cspace='RGB', orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = []
        if cspace != 'RGB':
            feature_image = dl.convert_color(image, conv=cspace)
        else: 
        	feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(dl.get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = dl.get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


# Feature extraction
def feature_extraction(cars, not_cars):
	
	# Main function internal parameters
	colorspace = 'RGB' # Can be RGB, LUV, YCrCb
	orient = 8
	pix_per_cell = 8
	cell_per_block = 4
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

	# feature extraction 
	car_features = extract_features(cars, cspace=colorspace, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel)

	notcar_features = extract_features(not_cars, cspace=colorspace, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel)

	return car_features, notcar_features

# Train module entry point
def train_classifier():

	# Get process time
	t=time.time()

	# Get list of files already shufled
	cars, not_cars  = get_data()
	print('Cars loaded:' + str(len(cars)))
	print('Not cars loaded:' + str(len(not_cars)))

	# feature extraction 
	car_features, notcar_features = feature_extraction(cars, not_cars)


