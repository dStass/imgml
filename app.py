import math
import os

import numpy as np
import cv2

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection, EntropyEdgeDetection, CannyEdgeDetection
from processing.model import Model
from processing.inpaint_processing import ImageInpainter

# # # # # # # # # # #
#   START OF FILE   #
# # # # # # # # # # #


# import image
IMG_PATH = 'assets/spaceman.jpg'

ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)

# image = cv2.imread(IMG_PATH)
loaded_images = io.load_recursive_quantised(IMG_PATH, 20, 2, 8)
# loaded_edges = [CannyEdgeDetection().generate_edges(im) for im in loaded_images]
loaded_edges = [EntropyEdgeDetection().generate_binary_edges_heatmap(im) for im in loaded_images]
current_display = loaded_images

# display settings
current_image_index = 0
show_edges_bool = False
FILL_COLOUR = (212, 12, 243)
BLACK_COLOUR = (0,0,0)
MOUSE_RADIUS = 6
MOUSE_ALPHA = 0.7

# edges_canny = cv2.Canny(np.copy(loaded_images[0]), threshold1=100, threshold2 = 200, apertureSize=3)
# image2 = io.load(IMG_PATH)
image_to_show = loaded_images[current_image_index]


# mouse states
mouse_pressed = False
mouse_x = -1
mouse_y = -1

# mouse events
def mouse_callback(event, x, y, flags, param):
  global image_to_show, mouse_pressed, mouse_x, mouse_y

  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_pressed = True
    s_x, s_y = x, y
    print(x, y)
    # image_to_show = cv2.rectangle(np.copy(current_display[current_image_index]), (0, 0), (60, 60), (0, 255, 0), 2)
    
    # image_to_show = np.copy(np.copy(loaded_images[0]))

  # elif event == cv2.EVENT_MOUSEMOVE:
  #   if not mouse_pressed:
  #     overlay = current_display[current_image_index].copy()
  #     alpha_value = 0.1

  #     # -1 to fill the circle
  #     cv2.circle(overlay, (x,y), MOUSE_RADIUS, FILL_COLOUR, -1) 
  #     image_to_show = cv2.addWeighted(overlay, alpha_value, image_to_show, 1 - alpha_value, 0)
  #     # cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

  elif event == cv2.EVENT_LBUTTONUP:
    mouse_pressed = False
    image_to_show = ImageInpainter().fill_image(current_display[current_image_index], loaded_edges[current_image_index], (y, x), MOUSE_RADIUS, BLACK_COLOUR, FILL_COLOUR)


  mouse_x = x
  mouse_y = y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)


while True:
  cv2.imshow('image', image_to_show)
  k = cv2.waitKey(1)

  if k == ord('c'):
    if s_y > e_y:
      s_y, e_y = e_y, s_y
    if s_x > e_x:
      s_x, e_x = e_x, s_x

    if e_y - s_y > 1 and e_x - s_x > 0:
      image = image[s_y:e_y, s_x:e_x]
      image_to_show = np.copy(image)
  
  # toggle edges
  elif k == ord('e'):
    if show_edges_bool: show_edges_bool = False
    else: show_edges_bool = True

    if show_edges_bool: current_display = loaded_edges
    else: current_display = loaded_images

    image_to_show = np.copy(np.copy(current_display[current_image_index]))

  # cycle through images
  # cycle left
  elif k == 123 or k == ord(','):
    if current_image_index > 0: current_image_index -= 1
    image_to_show = np.copy(np.copy(current_display[current_image_index]))
  
  # cycle right
  elif k == 124 or k == ord('.'):
    if current_image_index < len(loaded_images) - 1: current_image_index += 1
    image_to_show = np.copy(np.copy(current_display[current_image_index]))

  # pointer
  elif k == 45 or k == ord(']'):
    MOUSE_RADIUS = max(10, MOUSE_RADIUS + 1)

  elif k == 43 or k == ord('['):
    MOUSE_RADIUS = max(2, MOUSE_RADIUS - 1)

  # reset
  elif k == ord('r'):
    image_to_show = current_display[current_image_index]

  # exit
  elif k == 27 or k == ord('q'):
    break
  
  elif k != -1:
    print("Unsuported key : ", k)

  
  # every frame:

  # update cursor
  # overlay
  overlay = image_to_show.copy()

  # -1 to fill the circle
  cv2.circle(overlay, (mouse_x, mouse_y), MOUSE_RADIUS, FILL_COLOUR, -1) 
  image_to_show = cv2.addWeighted(overlay, MOUSE_ALPHA, image_to_show, 1 - MOUSE_ALPHA, 0)

cv2.destroyAllWindows()


# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #


# # declare paths
# path = 'assets/'
# outpath = 'output/'
# filename = 'spaceman.jpg'
# filepath = path + filename
# filename_split = filepath.split('.')
# name = filename_split[0]
# ext = filename_split[1]

# # reclare outpath
# outpath = outpath + name + '/'
# entropypath = outpath + 'entropy/'

# # handle directories
# # main directory
# if not os.path.exists(outpath):
#   os.makedirs(outpath)

# # handle entropy directory
# if not os.path.exists(entropypath):
#   os.makedirs(entropypath)

# # declare objects
# io = ImageIO(ext)
# e = EntropyEdgeDetection()


# START_BINS = 24
# END_BINS = 6
# STEP_BINS = 6

# PREV_BINS = START_BINS

# # save entropy layers
# entropy_mats = {}
# for end_bins in range(START_BINS, END_BINS - 1, -STEP_BINS):
#   # load and resize original image
#   img_mat = io.load_recursive_quantised(filepath, PREV_BINS, END_BINS, True)
#   io.save(img_mat, "loaded_img" + str(PREV_BINS) + '_' + str(end_bins), path=outpath)
#   PREV_BINS -= STEP_BINS

#   for kradius in range(1,4):
#     entropy_map = e.detect_edges(img_mat, kradius)
#     entropy_mat = e.generate_binary_edges_heatmap(img_mat, entropy_map, kradius, 0)

#     mat_key = 'bins' + str(end_bins) + '_kradius' + str(kradius)
#     entropy_mats[mat_key] = entropy_mat

#     # io.save(entropy_mat, "kradius" + str(kradius) + "_bins" + str(end_bins), path=entropypath)

# for key in entropy_mats:
#   entropy_mat = entropy_mats[key]
#   io.save(entropy_mat, str(key), path=entropypath)

# print(entropy_mats)
# final_mat = e.combine_img_mats(entropy_mats)
# io.save(final_mat, "FINAL", path=outpath)