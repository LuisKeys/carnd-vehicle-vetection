import cv2
import detection_library as dl
import matplotlib.pyplot as plt
import training_classifier as tc

# Train classifier
def traing_classifier():
	print('Train classifier')
	tc.train_classifier()

# Test Image
def test_image():

	# open image
	path = "./test_images/"
	file_name = "test1.jpg"
	image = cv2.imread(path + file_name)	

	# classify
	dl.detect(image)

	# display image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.show()


traing_classifier()
test_image()
