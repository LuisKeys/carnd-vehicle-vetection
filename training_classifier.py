import sys
import numpy as np
import glob
import detection_library as dl
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from random import shuffle

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
def extract_features(imgs, cspace='YCrCb', orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                     spatial_size=(32, 32), hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)

        # Depending on file extension mpimg.imread scale different
        # The following code is in place to prevent scale differences
        scale_factor = 1.0
        if file.endswith(('.jpeg', '.jpg')):
        	scale_factor = 255.0
        image = image.astype(np.float32) / scale_factor

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
        # Get color features
        spatial_features = dl.bin_spatial(feature_image, size=spatial_size)
        hist_features = dl.color_hist(feature_image, nbins=hist_bins)

        # Append the new feature vector to the features list
        all_features = np.hstack((spatial_features, hist_features, hog_features))
        features.append(all_features)


    # Return list of feature vectors
    return features


# Feature extraction
def feature_extraction(cars, not_cars):
	
	# Main function internal parameters
	colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins = dl.get_main_params()

	car_features = extract_features(cars, cspace=colorspace, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

	notcar_features = extract_features(not_cars, cspace=colorspace, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

	return car_features, notcar_features

def get_sets(car_features, notcar_features):
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Scale
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	# Labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split training and test sets
	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	return X_train, X_test, y_train, y_test, X_scaler

# Train module entry point
def train_classifier():

	# Get list of files already shufled
	cars, not_cars  = get_data()
	print('Cars loaded:' + str(len(cars)))
	print('Not cars loaded:' + str(len(not_cars)))

	# Feature extraction 
	car_features, notcar_features = feature_extraction(cars, not_cars)

	# Get sets
	X_train, X_test, y_train, y_test, X_scaler = get_sets(car_features, notcar_features)

	print('Feature vector length:', len(X_train[0]))

	# Create a linear SVC 
	parameters = {'C':[1, 10]}
	svc = LinearSVC()
	svc = GridSearchCV(svc, parameters)

	# Train model
	svc.fit(X_train, y_train)

	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	return X_scaler, svc