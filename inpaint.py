import time
import heapq

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from processing.image_processing import ImageIO
from processing.edge_processing import CannyEdgeDetection

# Functions

def normalised(nparr):
  norm = np.linalg.norm(nparr)
  max_val = np.amax(nparr)
  return nparr / max_val

def get_patch_points(patch, mask, edges, num_patch_elements):
  """
  Returns 2 things:
  1. points_size = number of "black" pixels in mask (or unmasked pixels)
     -> synonynomous to pixels that exist that we can extrapolate from
  2. points_edtes = sum of how many "white" pixels in edges image
     -> as edges and corners are white, we can use this as a guide to
        measure importance of a patch

  Assumptions:
  - RGB values are the same (binary image) in edges, we only extract the red
    pixel for simplicity
  """
  points_size = 0
  points_edges = 0
  for coordinate in patch:
    points_edges += (edges[coordinate[0]][coordinate[1]][0] * (1 / 255))
    if mask[coordinate[0]][coordinate[1]] == MASK_NONE: points_size += 1
  points_edges /= num_patch_elements
  points_size
  return (points_size, points_edges)

def get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements):
  to_return = []
  operation_count = 0

  # min and max masked elements in a patch arbitrarily chosen
  min_patch_size = int(0.3 * num_patch_elements)
  max_patch_size = num_patch_elements

  for fill_coordinates in to_fill:
    # optimisation: skip non-border points

    patches_in = coordinates_to_patch[fill_coordinates]

    each_patch = []
    for patch_id in patches_in:
      patch = patch_to_coordinates[patch_id]
      patch_information = get_patch_points(patch, mask, edges, num_patch_elements)
      patch_size = patch_information[0]
      patch_points = patch_information[1]
      if min_patch_size <= patch_size <= max_patch_size:
        heapq.heappush(each_patch, (-patch_points, -patch_size, patch_id))

    if len(each_patch) == 0: continue

    best_patch = heapq.heappop(each_patch)
    best_patch_points = best_patch[0]
    best_patch_size = best_patch[1]
    best_patch_id = best_patch[2]

    # use a heap as a queue to maintain importance of a particular point
    heapq.heappush(to_return, (best_patch_points, best_patch_size, best_patch_id))
    operation_count += 1

  return to_return

# Start of program

IMG_PATH = 'assets/spaceman.jpg'
MASK_PATH = 'output/space_mask.jpg'
OUT_FOLDER = 'output/patch/'
OUT_NAME = 'space_patch_'

# MASK_INPAINT = [255, 255, 255]
MASK_NONE = [0, 0, 0]

ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)

img_np = io.load(IMG_PATH)
edges = CannyEdgeDetection().generate_edges(img_np).tolist()

img = img_np.tolist()
mask_np = io.load(MASK_PATH)

mask_np[mask_np > 126] = 255
mask_np[mask_np <= 126] = 0

mask = mask_np.tolist()


PATCH_SIZE = 7
num_patch_elements = PATCH_SIZE * PATCH_SIZE
RESET_RATE = 0.20

# contained information
patch_to_coordinates = {}
coordinates_to_patch = {}
to_fill = set()

# choosing the importance of which point to fill
# format -> (no. surrounding unmasked, edge point, heap_operation_count, ..)
importance = []

patch_no = 0

# build patches 
# iterate over array
nrows = len(img)
ncols = len(img[0])
for row in range(nrows - PATCH_SIZE):
  for col in range(ncols - PATCH_SIZE):
    patch_to_coordinates[patch_no] = set()

    # iterate over PATCH_SIZE x PATCH_SIZE window
    for prow in range(PATCH_SIZE):
      for pcol in range(PATCH_SIZE):

        # sum (trow = total row, tcol = total col)
        trow = row + prow
        tcol = col + pcol
        coordinates = (trow, tcol)

        patch_to_coordinates[patch_no].add(coordinates)

        # only add point to partial map if it is a black pixel
        if mask[trow][tcol] != MASK_NONE: to_fill.add(coordinates)
        
        # build patch information into dicts
        if coordinates not in coordinates_to_patch: coordinates_to_patch[coordinates] = set()
        coordinates_to_patch[coordinates].add(patch_no)

     # increment patch
    patch_no += 1

# extract importance of patches
importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements)

steps = 0
while to_fill:
  steps += 1
  
  # re-calibrate when we get to a 
  if len(importance) == 0:
    importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements)
  
  important_tuple = heapq.heappop(importance)
  num_unmasked = important_tuple[1]
  patch_id = important_tuple[2]

  # this is the patch we will use
  identified_patch = patch_to_coordinates[patch_id]

  # deduce coordinates that need filling
  fill_patch = {c for c in identified_patch if mask[c[0]][c[1]] != MASK_NONE}

  # fill coordinates
  for coordinates in fill_patch:
    img[coordinates[0]][coordinates[1]] = [255, 0, 0]
    mask[coordinates[0]][coordinates[1]] = MASK_NONE

  # remove coordinates that have been filled and update our mask
  to_remove = set()
  for coordinates in fill_patch:
    to_fill.remove(coordinates)
    mask[coordinates[0]][coordinates[1]] = MASK_NONE

  print("len=", len(to_fill))
  
  # DEBUG: 
  # if steps % 50 == 0 or len(to_fill) == 0:
  #   io.save(img, OUT_NAME + str(steps), OUT_FOLDER, True)





