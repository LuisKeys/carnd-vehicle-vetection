import numpy as np
import cv2
import sys
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

this = sys.modules[__name__]
this.bboxes_per_frame = []
this.frame_counter = 0

# Cheks bbox with hitory in the last 3 frames
def check_history():
  status = False

  # If more than one frame, then walk through previous
  if len(this.bboxes_per_frame) > 2:
    for hist_frame1 in this.bboxes_per_frame[len(this.bboxes_per_frame) - 1]:
      for hist_frame2 in this.bboxes_per_frame[len(this.bboxes_per_frame) - 2]:
      # Get previous related bbox if any
        bbox, previous_bbox, status = check_proximity(hist_frame1, hist_frame2)
      # If a previous frame was found then continue process
      if status:
        # Get avg with previous bbox and current to smooth changes between frames
        return bbox, True

  return None, False


# Gets bbox centroid
def centroid(bbox):
  left, top = bbox[0]
  right, bottom = bbox[1]
  return (left + (right - left) // 2, top + (bottom - top) // 2)

# Checks bbox width / height relationship = shape aspect
def check_shape_aspect_position(bbox):
  left, top = bbox[0]
  right, bottom = bbox[1]
  width = right - left + 1
  height = bottom - top + 1

  if top > 420 and left > 1000:
    return False

  if top > 400 and left > 1100:
    return False

  if width / height <=1.7:
    if width / height >=0.8:
      return True

  return False

# Check if a label is related with a previous frame label
def check_proximity(bbox, hist_frame):

  previous_bbox = bbox
  status = True
  max_x_distance = 90.0
  max_y_distance = 90.0

  if len(hist_frame) == 0:
    return bbox, previous_bbox, True

  if abs(centroid(bbox)[0] - centroid(hist_frame)[0]) < max_x_distance:
    if abs(centroid(bbox)[1] - centroid(hist_frame)[1]) < max_y_distance:
      status = True
      previous_bbox = hist_frame

  return bbox, previous_bbox, status

# Get AVG betweem 2 bbox
def get_avg_bbox(bbox, previous_bbox):
  left, top = bbox[0]
  right, bottom = bbox[1]

  prev_left, prev_top = previous_bbox[0]
  prev_right, prev_bottom = previous_bbox[1]

  previous_weight = 0.8

  avg_left = int((previous_weight * prev_left + (1.0 - previous_weight) * left))
  avg_right = int((previous_weight * prev_right + (1.0 - previous_weight) * right))
  avg_top = int((previous_weight * prev_top + (1.0 - previous_weight) * top))
  avg_bottom = int((previous_weight * prev_bottom + (1.0 - previous_weight) * bottom))

  avg_bbox = ((avg_left, avg_top), (avg_right, avg_bottom))

  return avg_bbox

# Process bboxes based on previous frames
# to provide more stability and reliability
def process_bbox(bbox):
  
  previous_bbox = bbox  
  status = False

  # No history, then assign defaults
  if len(this.bboxes_per_frame) == 0:
    return bbox

  # If more than one frame, then walk through previous
  for hist_frame in this.bboxes_per_frame[len(this.bboxes_per_frame) - 1]:
    # Get previous related bbox if any
    bbox, previous_bbox, status = check_proximity(bbox, hist_frame)
    # If a previous frame was found then continue process
    if status:
      # Get avg with previous bbox and current to smooth changes between frames
      bbox = get_avg_bbox(bbox, previous_bbox)

  # Return defaults
  return bbox


# Provides main process hyper params
def get_main_params():

  # Main function internal parameters
  colorspace = 'YCrCb' # Can be RGB, LUV, YCrCb
  orient = 9
  pix_per_cell = 8
  cell_per_block = 4
  hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
  spatial_size = (32, 32)
  hist_bins = 32

  return colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins

# Threshold for heatmap
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Add heat map
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for bbox in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        left, top, right, bottom = bbox
        heatmap[top:bottom, left:right] += 1

    # Return updated heatmap
    return heatmap

#Draw labeled boundary boxes based on heat map and resulting labels
def draw_labeled_bboxes(img, labels):
    final_bboxes = []
    draw_rectangles = False
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        final_bboxes.append(bbox)
        # Draw the box on the image
        bbox = process_bbox(bbox)
        # Only draw bbox if it is valie
        if check_shape_aspect_position(bbox):
          draw_rectangles = True
          cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)

    if draw_rectangles == False:
      bbox, status = check_history()
      if status:
        draw_rectangles = True
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
  
    #Add final bboxes to bboxes per frame list to keep history of all previous frames
    this.bboxes_per_frame.append(final_bboxes)
    # Return the image
    return img, draw_rectangles

# Color conversion 
def convert_color(img, from_space='RGB', to_space='YCrCb'):
  if from_space=='RGB':
    if to_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if to_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

  if from_space=='BGR':
    if to_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if to_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

# Histogram of oriented gradients (HOG) features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)

        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Spatial features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Color histogram features                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, color_from, color_to, scale_factor, 
              start_y, stop_y, start_x, stop_x, scale, svc, X_scaler, 
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32) / scale_factor
    
    img_tosearch = img[start_y:stop_y,start_x:stop_x,:]
    ctrans_tosearch = convert_color(img_tosearch, from_space=color_from, to_space=color_to)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # holds boundary box list
    bbox_list = []
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps // 2, nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            all_features = np.hstack((spatial_features, hist_features, hog_features))
    
            transf_features = all_features.reshape(1, -1)
    
            test_features = X_scaler.transform(transf_features)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox = (xbox_left, ytop_draw+start_y, xbox_left+win_draw, ytop_draw+win_draw+start_y)
                left, top, right, bottom = bbox
                bbox_list.append(bbox)
                # cv2.rectangle(draw_img,(left, top),(right,bottom),(0,0,255),6) 

    # Create heatmap
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, 1)
    heatmap = np.clip(heatmap, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    # mpimg.imsave("./examples/frame_" + str(this.frame_counter) + ".png", draw_img)

    draw_img = draw_labeled_bboxes(draw_img, labels)

    # mpimg.imsave("./examples/heatmap_" + str(this.frame_counter) + ".png", heatmap)

    return draw_img

# Entry point of detection module
def detect(image, svc, X_scaler, scale_factor, color_from = 'BGR', color_to = 'YCrCb'):

  # Get main hyper params common to trainnig and prediciton
  colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins = get_main_params()

  scale = 1.5

  start_y = 400
  stop_y = 600
  start_x = 0
  stop_x = 1280

  this.frame_counter += 1

  # Detect cars using sub-sampling windows search
  out_img, draw_rectangles = find_cars(image, color_from, color_to, scale_factor, start_y, 
                      stop_y, start_x, stop_x, scale, svc, X_scaler, orient, pix_per_cell, 
                      cell_per_block, spatial_size, hist_bins)

  # Representation of region to scan
  # cv2.rectangle(out_img, (start_x, start_y), (stop_x, stop_y), (255,0,255), 6)

  return out_img
