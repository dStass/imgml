import heapq
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from processing.image_processing import ImageIO
from processing.edge_processing import CannyEdgeDetection

# Functions

def get_neighbours(coordinates, img):
  """
  get surrounding neighbours to a (r, c) pair in a given img
  handles extreme image edge cases
  """
  nrows = len(img)
  ncols = len(img[0])
  r = coordinates[0]
  c = coordinates[1]
  to_return = [
    [r - 1, c],
    [r + 1, c],
    [r, c - 1],
    [r, c + 1]
  ]
  return [t for t in to_return if 0 <= t[0] < nrows and 0 <= t[1] < ncols]


def get_squared_difference(arr1, arr2):
  """
  assumes len(arr1) == len(arr2)
  """
  to_return = 0
  for i in range(len(arr1)):
    to_return += pow((arr1[i] - arr2[i]), 2)
  return to_return


def normalised(nparr):
  norm = np.linalg.norm(nparr)
  max_val = np.amax(nparr)
  return nparr / max_val

def get_patch_points(patch, mask, edges, num_patch_elements):
  """
  Returns 2 things:
  1. points_size = number of "black" pixels in mask (or unmasked pixels)
     -> synonynomous to pixels that exist that we can extrapolate from
  2. points_edges = sum of how many "white" pixels in edges image
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

def get_patch_masked_num(patch, mask):
  """
  Returns number of unmasked pixels
  """
  unmasked = 0
  for coordinate in patch:
    if mask[coordinate[0]][coordinate[1]] != MASK_NONE: unmasked += 1
  return unmasked

def get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements):
  """
  traverses to_fill (points that needs to be inpainted)
  """
  to_return = []
  operation_count = 0

  # min and max masked elements in a patch arbitrarily chosen
  min_patch_size = int(0.5 * num_patch_elements)
  max_patch_size = num_patch_elements

  for fill_coordinates in to_fill:
    # optimisation: skip non-border points
    neighbours = [n for n in get_neighbours(fill_coordinates, mask) if mask[n[0]][n[1]] == MASK_NONE]
    if len(neighbours) == 0: continue
    
    # extract patches this coordinate appears in
    patches_appear_in = coordinates_to_patch[fill_coordinates]

    # consider each of these patches and measure each patch by importance
    viable_patches = []
    for patch_id in patches_appear_in:
      patch = patch_to_coordinates[patch_id]
      patch_information = get_patch_points(patch, mask, edges, num_patch_elements)
      patch_size = patch_information[0]
      patch_points = patch_information[1]
      if min_patch_size <= patch_size <= max_patch_size:
        heapq.heappush(viable_patches, (-patch_points, -patch_size, patch_id))

    # if no solution can be found, we skip this coordinate
    if len(viable_patches) == 0: continue

    best_patch = heapq.heappop(viable_patches)
    best_patch_points = best_patch[0]
    best_patch_size = best_patch[1]
    best_patch_id = best_patch[2]

    # use a heap as a queue to maintain importance of a particular point
    heapq.heappush(to_return, (best_patch_points, best_patch_size, best_patch_id))
    operation_count += 1

  return to_return

def get_similar_patch(identified_patch_id, img, mask, edges):
  """


  """

  MAX_SEARCH_RADIUS = 32
  MIN_SEARCH_RADIUS = 2

  identified_patch = patch_to_coordinates[identified_patch_id]
  identified_topleft = identified_patch[0]

  ncoordinates = len(identified_patch)  # extract the length of elements in identified_patch
  candidate_patches = []
  for candidate_patch_id in full_patches:
    # if candidate_patch_id == 
    candidate_patch = patch_to_coordinates[candidate_patch_id]
    candidate_topleft = candidate_patch[0]

    distance_from_identified = math.sqrt(get_squared_difference(identified_topleft, candidate_topleft))
    if distance_from_identified < MIN_SEARCH_RADIUS or distance_from_identified > MAX_SEARCH_RADIUS: continue



    # compare each pair of coordinates
    # sum on squared differences
    points_rgb = 0
    available_information = 0
    for index in range(ncoordinates):
      identified_coordinate = identified_patch[index]
      identified_row = identified_coordinate[0]
      identified_col = identified_coordinate[1]

      candidate_coordinate = candidate_patch[index]
      candidate_row = candidate_coordinate[0]
      candidate_col = candidate_coordinate[1]

      # skip comparison if it is a point we are trying to fill
      if mask[identified_row][identified_col] != MASK_NONE:
        continue
      
      points_rgb += get_squared_difference(img[identified_row][identified_col], img[candidate_row][candidate_col])
      available_information += 1

    points = points_rgb
    heapq.heappush(candidate_patches, (points, available_information, candidate_patch_id))
  
  most_similar_tup = heapq.heappop(candidate_patches)

  most_similar = most_similar_tup[-1]
  print(identified_patch[0], patch_to_coordinates[most_similar][0])
  return most_similar



# Start of program

IMG_PATH = 'assets/spaceman.jpg'
MASK_PATH = 'output/space_mask.jpg'
OUT_FOLDER = 'output/inpaint/'
OUT_NAME = 'space_patch_'


# image reader
io = ImageIO(ext=IMG_PATH.split('.')[1])

# read image
img_np = io.load(IMG_PATH)
img = img_np.tolist()

# build edges
edges = CannyEdgeDetection().generate_edges(img_np).tolist()

# build mask
MASK_NONE = [0, 0, 0]
mask_np = io.load(MASK_PATH)
mask_np[mask_np > 126] = 255
mask_np[mask_np <= 126] = 0
mask = mask_np.tolist()

# patch information
PATCH_SIZE = 7
num_patch_elements = PATCH_SIZE * PATCH_SIZE

# contained information
patch_to_coordinates = {}  # maps a patch to coordinates that live inside it
coordinates_to_patch = {}  # maps a coordinate to all patches it exists in

# identifier mappings
identifier_patch_to_coordinates = {}  # maps a patch_id to its top left-corner coordinate
identifier_coordinates_to_patch = {}  # maps a coordinate to the patch where it is the most top left coordinate (if exists)

# sets
to_fill = set()  # set containing coordinates that require filling
full_patches = set()  # set containing ids of patches that are completely filled
partial_patches = set()  # set containing ids of patches that are not completely filled

patch_id = 0

# build patches 
# iterate over array
nrows = len(img)
ncols = len(img[0])
for row in range(nrows - PATCH_SIZE):
  for col in range(ncols - PATCH_SIZE):
    patch_to_coordinates[patch_id] = []
    full_patches.add(patch_id)  # remove partial patches after

    # do identifier mappings
    identifier_coordinates_to_patch[(row, col)] = patch_id
    identifier_patch_to_coordinates[patch_id] = (row, col)

    # iterate over PATCH_SIZE x PATCH_SIZE window
    for prow in range(PATCH_SIZE):
      for pcol in range(PATCH_SIZE):

        # sum (trow = total row, tcol = total col)
        trow = row + prow
        tcol = col + pcol
        coordinates = (trow, tcol)

        patch_to_coordinates[patch_id].append(coordinates)

        # add coordinates to to_fill if it doesn't have a black mask pixel
        # i.e. this pixel is required to be inpainted
        if mask[trow][tcol] != MASK_NONE:
          to_fill.add(coordinates)
          partial_patches.add(patch_id)
        
        # build patch information into dicts
        if coordinates not in coordinates_to_patch: coordinates_to_patch[coordinates] = []
        coordinates_to_patch[coordinates].append(patch_id)

     # increment patch
    patch_id += 1

# remove partial patches
for patch_id in partial_patches:
  full_patches.remove(patch_id)

# choosing the importance of which point to fill
# format -> (no. surrounding unmasked, edge point, heap_operation_count, ..)
# extract importance of patches
importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements)

# begin inpatinging
steps = 0
while to_fill:
  steps += 1
  
  # re-calibrate important points when required
  if len(importance) == 0:
    importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, edges, num_patch_elements)
  
  # extract patch to be inpainted
  important_tuple = heapq.heappop(importance)
  num_unmasked = important_tuple[1]

  # this is the patch we will use
  identified_patch_id = important_tuple[2]
  identified_patch = patch_to_coordinates[identified_patch_id]

  # early exit for repeated work
  if identified_patch_id in full_patches:
    continue

  # find the most similar patch to identified_patch
  similar_patch_id = get_similar_patch(identified_patch_id, img, mask, edges)
  similar_patch = patch_to_coordinates[similar_patch_id]

  # fill coordinates
  # for coordinates in coordinates_to_fill:
  #   img[coordinates[0]][coordinates[1]] = [255, 0, 0]
  #   mask[coordinates[0]][coordinates[1]] = MASK_NONE

  for index in range(len(identified_patch)):
    identified_coordinate = identified_patch[index]
    identified_row = identified_coordinate[0]
    identified_col = identified_coordinate[1]

    similar_coordinate = similar_patch[index]
    similar_row = similar_coordinate[0]
    similar_col = similar_coordinate[1]

    # fill img if it is a point we are trying to fill
    if mask[identified_row][identified_col] != MASK_NONE:
      img[identified_row][identified_col] = img[similar_row][similar_col]

  # update full and partial patches
  partial_patches.remove(identified_patch_id)
  full_patches.add(identified_patch_id)

  # remove coordinates that have been filled and update our mask
  to_remove = set()
  coordinates_to_fill = {c for c in identified_patch if mask[c[0]][c[1]] != MASK_NONE}
  for coordinates in coordinates_to_fill:
    to_fill.remove(coordinates)
    mask[coordinates[0]][coordinates[1]] = MASK_NONE

    # connected patches to filled coordinates
    for connected_patch_id in coordinates_to_patch[coordinates]:
      connected_patch = patch_to_coordinates[connected_patch_id]

      # if the connected patch is now filled
      if connected_patch_id in full_patches: continue
      if get_patch_masked_num(connected_patch, mask) == 0:
        if connected_patch_id in partial_patches:
          partial_patches.remove(connected_patch_id)
          full_patches.add(connected_patch_id)


  print("len=", len(to_fill))
  
  # DEBUG: 
  if steps % 10 == 0 or len(to_fill) == 0:
    io.save(img, OUT_NAME + str(steps), OUT_FOLDER, True)
print()