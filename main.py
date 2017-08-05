import sys
import cv2
import detection_library as dl
import matplotlib.pyplot as plt
import training_classifier as tc
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

this = sys.modules[__name__]

# Train classifier
def traing_classifier():
	print('Train classifier')
	this.X_scaler, this.svc  = tc.train_classifier()

# Video Process callback function
def process_image(image):

	scale_factor = 255.0
	# classify
	out_image = dl.detect(image, this.svc, this.X_scaler, scale_factor,
						  color_from='RGB', color_to='YCrCb')
	return out_image

# Video Process
def test_video():
	# Load trained svc and scaler
	this.svc = joblib.load('svc_lin_c10.pkl')
	this.X_scaler = joblib.load('X_scaler.pkl')
	# Videos paths:
	video_input = './project_video.mp4'
	video_output = './project_video_output.mp4'
	clip = VideoFileClip(video_input).subclip(40.5, 41)
	#clip = VideoFileClip(video_input)
	white_clip = clip.fl_image(process_image)
	# Save output video
	white_clip.write_videofile(video_output, audio=False)
 

# Test Image
def test_image():

	# open image
	path = "./test_images/"
	file_name = "test1.jpg"
	image = cv2.imread(path + file_name)	

	scale_factor = 255.0
	this.svc = joblib.load('svc_lin_c10.pkl')
	this.X_scaler = joblib.load('X_scaler.pkl')
	# classify
	out_image = dl.detect(image, this.svc, this.X_scaler, scale_factor,
						  color_from='BGR', color_to='YCrCb')

	# display image
	out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

	plt.imshow(out_image)
	plt.show()


#traing_classifier()
#test_image()
test_video()
