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
  points_size = 0
  points_edges = 0
  for coordinate in patch:
    points_edges += (edges[coordinate[0]][coordinate[1]][0] * (1 / 255))
    if mask[coordinate[0]][coordinate[1]] == MASK_NONE: points_size += 1
  points_edges /= num_patch_elements
  points_size /= num_patch_elements
  points = points_size + points_edges
  return points

def get_patch_points_size(patch, mask, edges, num_patch_elements):
  points_size = 0
  for coordinate in patch:
    if mask[coordinate[0]][coordinate[1]] == MASK_NONE: points_size += 1
  points_size
  return points_size

def get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, partial_patch_to_coordinates, edges, num_patch_elements):
  # best_points = -1
  # best_id = None
  to_return = []
  operation_count = 0
  for fill_coordinates in to_fill:
    patches_in = coordinates_to_patch[fill_coordinates]

    each_patch = []
    for patch_id in patches_in:
      # partial_patch = partial_patch_to_coordinates[patch_id]
      # identified_patch = patch_to_coordinates[patch_id]
      all_patch = patch_to_coordinates[patch_id]
      # full_patch = {p for p in all_patch if p not in partial_patch}
      # combined_patch = full_patch or partial_patch

      patch_points = get_patch_points(all_patch, mask, edges, num_patch_elements)
      patch_size = get_patch_points_size(all_patch, mask, edges, num_patch_elements)
      heapq.heappush(each_patch, (-patch_size, -patch_points, patch_id))
  
    best_patch = heapq.heappop(each_patch)
    best_patch_size = best_patch[0]
    best_patch_points = best_patch[1]
    best_patch_id = best_patch[2]

    # use a heap as a queue to maintain importance of a particular point
    heapq.heappush(to_return, (best_patch_size, best_patch_points, best_patch_id))
    operation_count += 1

    # to_return.append((best_patch_points, operation_count, fill_coordinates, best_patch_id))
    # if best_patch_points > best_points:
    #   best_points = best_patch_points
    #   best_id = best_patch_id
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
RESET_RATE = 0.005

# contained information
patch_to_coordinates = {}
partial_patch_to_coordinates = {}  # contains non-mask pixels only
coordinates_to_patch = {}
to_fill = set()

# choosing the importance of which point to fill
# format -> (no. surrounding unmasked, edge point, heap_operation_count, ..)
importance = []

patch_no = 0
heap_operation_count = 0  # used for heap operation to maintain stability

# build patches 
# iterate over array
nrows = len(img)
ncols = len(img[0])
for row in range(nrows - PATCH_SIZE):
  for col in range(ncols - PATCH_SIZE):
    patch_to_coordinates[patch_no] = set()
    partial_patch_to_coordinates[patch_no] = set()
    # patch_to_importance = 0

    # iterate over PATCH_SIZE x PATCH_SIZE window
    for prow in range(PATCH_SIZE):
      for pcol in range(PATCH_SIZE):

        # sum (trow = total row, tcol = total col)
        trow = row + prow
        tcol = col + pcol
        coordinates = (trow, tcol)

        patch_to_coordinates[patch_no].add(coordinates)

        # only add point to partial map if it is a black pixel
        if mask[trow][tcol] == MASK_NONE: partial_patch_to_coordinates[patch_no].add(coordinates)
        else: to_fill.add(coordinates)        
        
        # build patch information into dicts
        if coordinates not in coordinates_to_patch: coordinates_to_patch[coordinates] = set()
        coordinates_to_patch[coordinates].add(patch_no)

     # increment patch
    patch_no += 1

# calculate initial importance
# for fill_coordinates in to_fill:
#   patches_in = coordinates_to_patch[fill_coordinates]
#   best_patch_points = -1
#   best_patch_id = None
#   for patch_id in patches_in:
#     partial_patch = partial_patch_to_coordinates[patch_id]
#     patch_points = get_patch_points(partial_patch, edges, num_patch_elements)
#     if patch_points > best_patch_points:
#       best_patch_points = patch_points
#       best_patch_id = patch_id

#   # use a heap as a queue to maintain importance of a particular point
#   heapq.heappush(importance, (-best_patch_points, heap_operation_count, fill_coordinates, best_patch_id))
#   heap_operation_count += 1

importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, partial_patch_to_coordinates, edges, num_patch_elements)

# deduce patches that are full (we only use these OR partial patches to fill)
# full_patch_to_coordinates = {patch_to_coordinates[c] for c in patch_to_coordinates if c not in partial_patch_to_coordinates}

reset_heap_every = int(RESET_RATE * len(to_fill))

last_reset = len(to_fill)
# traverse to_fill
steps = 0
while to_fill:
  steps += 1
  if steps == 3250:
    print()
  if steps == 3300:
    print()
  if len(to_fill) - last_reset > reset_heap_every:
    last_reset = len(to_fill)
    importance = get_important_patches(to_fill, mask, coordinates_to_patch, patch_to_coordinates, partial_patch_to_coordinates, edges, num_patch_elements)
  # find most important patch
  # patch_id = get_important_patches(to_fill, coordinates_to_patch, partial_patch_to_coordinates, edges, num_patch_elements)
  patch_id = heapq.heappop(importance)[2]
  # this is the patch we will use
  identified_patch = patch_to_coordinates[patch_id]
  # if patch_id not in partial_patch_to_coordinates: continue
  # partial_patch = partial_patch_to_coordinates[patch_id]
  # if len(partial_patch) == num_patch_elements: continue

  # deduce coordinates that need filling
  fill_patch = {c for c in identified_patch if mask[c[0]][c[1]] != MASK_NONE}

  # fill coordinates
  for coordinates in fill_patch:
    img[coordinates[0]][coordinates[1]] = [255, 0, 0]    

  # remove coordinates that have been filled
  to_remove = set()
  for coordinates in fill_patch:
    # to_remove.add(coordinates)
    to_fill.remove(coordinates)
    mask[coordinates[0]][coordinates[1]] = MASK_NONE
    connected_patches = coordinates_to_patch[coordinates]
    for patch_id in connected_patches:
      connected_patch = patch_to_coordinates[patch_id]

    #   connected_partial_patch.add(coordinates)
    #   full_patch_to_coordinates
      # if len(connected_partial_patch) == num_patch_elements:
      #   del partial_patch_to_coordinates[patch_id]


  img[coordinates[0]][coordinates[1]]
  print("len=", len(to_fill))
  
  if steps % 200 == 0:
    # plt.imshow(np.array(mask))
    # plt.show()
    io.save(img, OUT_NAME + str(steps), OUT_FOLDER, True)

  # print()
  # # extract info
  # # to_explore = importance.pop()
  # # fill_coordinates = to_explore[2]
  # # patch_id = to_explore[3]

  # # remove current coordinate from what we need to fill
  # to_fill.remove(fill_coordinates)

  # # this is the patch we will use
  # identified_patch = patch_to_coordinates[patch_id]
  # partial_patch = partial_patch_to_coordinates[patch_id]

  # # deduce coordinates that need filling
  # fill_patch = {c for c in identified_patch if c not in partial_patch}
  


  # print()









# for tups in importance:
#   print("points=", tups[0], "ncount=", len( partial_patch_to_coordinates[tups[3]] )) 

# R, G, B = cv2.split(img)

# R = R.astype('float64')
# G = G.astype('float64')
# B = B.astype('float64')

# R *= (1/255)
# G *= (1/255)
# B *= (1/255)

# eps = 0.00000000005
