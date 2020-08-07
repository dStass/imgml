# # # # # # # # # # #
#   START OF FILE   #
# # # # # # # # # # #

import math
import os

import numpy as np
import cv2

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection, EntropyEdgeDetection, CannyEdgeDetection
from processing.model import Model
from processing.inpaint_processing import ImageInpainter

# import image
IMG_PATH = 'assets/spaceman.jpg'

ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)

layers = []

# load layers
layers.append(io.load_recursive_quantised(IMG_PATH, 20, 2, 8))  # photos, edges, etc.
layers.append([CannyEdgeDetection().generate_edges(im) for im in layers[0]])
# loaded_edges = [EntropyEdgeDetection().generate_binary_edges_heatmap(im) for im in loaded_images]

# display settings
layer_index = 0
display_index = 0
current_layer = [np.copy(l) for l in layers[layer_index]]
current_display = current_layer[display_index]

# cursor settings
show_edges_bool = False
FILL_COLOUR = (212, 12, 243)
BLACK_COLOUR = (0,0,0)
MOUSE_RADIUS = 6
MOUSE_ALPHA = 0.7
mouse_pressed = False
mouse_x = -1
mouse_y = -1

# mouse events
def mouse_callback(event, x, y, flags, param):
  global current_display, mouse_pressed, mouse_x, mouse_y

  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_pressed = True
    # s_x, s_y = x, y
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
    # current_display
    # image_to_show = ImageInpainter().fill_image(current_display[current_image_index], loaded_edges[current_image_index], (y, x), MOUSE_RADIUS, BLACK_COLOUR, FILL_COLOUR)


  mouse_x = x
  mouse_y = y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)


while True:
  cv2.imshow('image', current_display)
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
  elif k == 123 or k == ord(','):
    if display_index > 0: display_index -= 1
    current_display = np.copy(np.copy(current_layer[display_index]))
  
  # cycle right
  elif k == 124 or k == ord('.'):
    if display_index < len(current_layer) - 1: display_index += 1
    current_display = np.copy(np.copy(current_layer[display_index]))

  # pointer
  elif k == 45 or k == ord(']'):
    MOUSE_RADIUS = max(10, MOUSE_RADIUS + 1)

  elif k == 43 or k == ord('['):
    MOUSE_RADIUS = max(2, MOUSE_RADIUS - 1)

  # reset
  elif k == ord('r'):
    current_display = current_layer[display_index]

  # exit
  elif k == 27 or k == ord('q'):
    break
  
  elif k != -1:
    print("Unsuported key : ", k)

  
  # every frame:

  # update cursor
  # overlay
  overlay = current_layer[display_index].copy()

  # -1 to fill the circle
  cv2.circle(overlay, (mouse_x, mouse_y), MOUSE_RADIUS, FILL_COLOUR, -1) 
  current_display = cv2.addWeighted(overlay, MOUSE_ALPHA, current_display, 1 - MOUSE_ALPHA, 0)

cv2.destroyAllWindows()


# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #