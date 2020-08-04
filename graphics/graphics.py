# imports

# system imports
import sys

currpkg = 'graphics'

for s in sys.path:
  if s.endswith(currpkg):
    sys.path.append(s[:-len(currpkg)])
    break

# packages imports
import numpy as np
import cv2

# intra-package imports
from processing.image_processing import ImageIO

# import image
IMG_PATH = 'assets/bridge.jpg'
image = cv2.imread(IMG_PATH)
image_to_show = np.copy(image)

# mouse states
mouse_pressed = False
s_x = s_y = e_x = e_y = -1

# mouse events
def mouse_callback(event, x, y, flags, param):
  global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_pressed = True
    s_x, s_y = x, y
    image_to_show = np.copy(image)

  elif event == cv2.EVENT_MOUSEMOVE:
    if mouse_pressed:
      image_to_show = np.copy(image)
      cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

  elif event == cv2.EVENT_LBUTTONUP:
    mouse_pressed = False
    e_x, e_y = x, y

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
  elif k == 27 or k == ord('q'):
    break

cv2.destroyAllWindows()