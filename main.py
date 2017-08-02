import sys
import cv2
import detection_library as dl
import matplotlib.pyplot as plt
import training_classifier as tc

this = sys.modules[__name__]

# Train classifier
def traing_classifier():
	print('Train classifier')
	this.X_scaler, this.svc  = tc.train_classifier()

# Test Image
def test_image():

	# open image
	path = "./test_images/"
	file_name = "test1.jpg"
	image = cv2.imread(path + file_name)	

	scale_factor = 255.0

	# classify
	out_image = dl.detect(image, this.svc, this.X_scaler, scale_factor,
						  color_from='BGR', color_to='YCrCb')

	# display image
	out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

	plt.imshow(out_image)
	plt.show()


traing_classifier()
test_image()
