# # # # # # # # # # #
#   START OF FILE   #
# # # # # # # # # # #

import math
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection, EntropyEdgeDetection, CannyEdgeDetection
from processing.model import Model
from processing.inpaint_processing import ImageInpainter

# Export settings
OUTPUT_FOLDER = "output/"
OUTPUT_NAME = "test"

# constants
NUM_LAYERS = 2  # layers per image

# import image
IMG_PATH = 'assets/car.jpg'
ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)
loaded_images = io.load_recursive_quantised(IMG_PATH, 20, 2, 8)

# set up images and layers
images = [[li] for li in loaded_images]

for i in range(len(images)):
  layer = images[i]
  layer.append(CannyEdgeDetection().generate_edges(layer[0]))

NUM_LAYERS = len(images[0])

# display settings
image_index = 0
layer_index = 0
current_layers = [np.copy(l) for l in images[image_index]]
current_display = current_layers[layer_index]
current_mask = np.zeros((loaded_images[0].shape)).astype('uint8')

# cursor settings
show_edges_bool = False
FILL_COLOUR = (212, 12, 243)
BLACK_COLOUR = (0, 0, 0)
WHITE_COLOUR = (255, 255, 255)
MOUSE_RADIUS = 6
MOUSE_ALPHA = 0.7
mouse_pressed = False
mouse_x = -1
mouse_y = -1

# mouse events
def mouse_callback(event, x, y, flags, param):
  def draw_circle():
    alpha_value = 1

    # -1 to fill the circle
    for i, layer in enumerate(current_layers):
      overlay = layer.copy()
      cv2.circle(overlay, (x,y), MOUSE_RADIUS, FILL_COLOUR, -1) 
      # image_to_show = cv2.addWeighted(overlay, alpha_value, image_to_show, 1 - alpha_value, 0)
      current_layers[i] = cv2.addWeighted(overlay, alpha_value, layer, 1 - alpha_value, 0)
      current_display = current_layers[i]
    
    # update mask
    global current_mask
    overlay = current_mask.copy()
    cv2.circle(overlay, (x,y), MOUSE_RADIUS, WHITE_COLOUR, -1)
    current_mask = cv2.addWeighted(overlay, alpha_value, current_mask, 1 - alpha_value, 0).astype('uint8')

  global current_display, mouse_pressed, mouse_x, mouse_y

  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_pressed = True
    draw_circle()
    # s_x, s_y = x, y
    print(x, y)
    # image_to_show = cv2.rectangle(np.copy(current_display[current_image_index]), (0, 0), (60, 60), (0, 255, 0), 2)
    
    # image_to_show = np.copy(np.copy(loaded_images[0]))

  elif event == cv2.EVENT_MOUSEMOVE:
    if mouse_pressed:
      draw_circle()
      # cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

  elif event == cv2.EVENT_LBUTTONUP:
    mouse_pressed = False

  mouse_x = x
  mouse_y = y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)


while True:
  # cv2.imshow('image', np.flip(current_mask, 2))
  cv2.imshow('image', np.flip(current_display, 2))
  k = cv2.waitKey(1)

  if k == ord('c'):
    # if s_y > e_y:
    #   s_y, e_y = e_y, s_y
    # if s_x > e_x:
    #   s_x, e_x = e_x, s_x

    # if e_y - s_y > 1 and e_x - s_x > 0:
    #   image = image[s_y:e_y, s_x:e_x]
    #   image_to_show = np.copy(image)
    pass 
  
  # toggle edges
  # elif k == ord('e'):
  #   if show_edges_bool: show_edges_bool = False
  #   else: show_edges_bool = True

  #   if show_edges_bool: current_display = loaded_edges
  #   else: current_display = loaded_images

  #   image_to_show = np.copy(np.copy(current_display[current_image_index]))

  # cycle through images
  # cycle left

  # handle layers
  elif k == ord('l'):
    layer_index += 1
    layer_index %= NUM_LAYERS
    # current_layers = [np.copy(l) for l in images[image_index]]
    current_display = current_layers[layer_index]

  elif k == 123 or k == ord(','):
    if image_index > 0: image_index -= 1
    current_layers = [np.copy(l) for l in images[image_index]]
    current_display = current_layers[layer_index]
    current_mask = np.zeros((loaded_images[0].shape)).astype('uint8')
  
  # cycle right
  elif k == 124 or k == ord('.'):
    if image_index < len(images) - 1: image_index += 1
    current_layers = [np.copy(l) for l in images[image_index]]
    current_display = current_layers[layer_index]
    current_mask = np.zeros((loaded_images[0].shape)).astype('uint8')

  # pointer
  elif k == 45 or k == ord(']'):
    MOUSE_RADIUS = max(10, MOUSE_RADIUS + 1)

  elif k == 43 or k == ord('['):
    MOUSE_RADIUS = max(2, MOUSE_RADIUS - 1)

  # reset
  elif k == ord('r'):
    current_layers = [np.copy(l) for l in images[image_index]]
    current_display = current_layers[layer_index]
    current_mask = np.zeros((loaded_images[0].shape)).astype('uint8')


  # inpaint
  elif k == ord('i'):
    inpainter = ImageInpainter()
    inpainter.remove_and_inpaint(OUTPUT_FOLDER, OUTPUT_NAME, images[image_index][0], current_mask)


    pass

  # exit
  elif k == 27 or k == ord('q'):
    break
  
  elif k != -1:
    print("Unsuported key : ", k)

  
  # every frame:

  # update cursor
  # overlay
  overlay = current_layers[layer_index].copy()

  # -1 to fill the circle
  cv2.circle(overlay, (mouse_x, mouse_y), MOUSE_RADIUS, FILL_COLOUR, -1) 
  current_display = cv2.addWeighted(overlay, MOUSE_ALPHA, current_display, 1 - MOUSE_ALPHA, 0)

cv2.destroyAllWindows()


# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #