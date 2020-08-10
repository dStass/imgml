import heapq
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from processing.model import LinearRegressionModel
from processing.image_processing import ImageIO
from processing.edge_processing import CannyEdgeDetection


class ImageInpainter:
  def __init__(self):
    pass

  def fill_image(self, current_image, edge_image, position, radius, from_colour, to_colour):
    fill_coordinates = []
    to_colour = list(to_colour)
    from_colour = list(from_colour)
    seen = {}
    def fill(current_mat, img_mat):
      while fill_coordinates:
        # curr_pair = fill_coordinates.pop(0)

        curr = fill_coordinates.pop(0)
        curr_row = curr[0]
        curr_col = curr[1]

        seen[tuple(curr)] = True


        img_mat[curr_row][curr_col] = to_colour
        current_mat[curr_row][curr_col] = to_colour
        neighbours = get_neighbouring_coordinates(curr)

        for n in neighbours:
          if tuple(n) in seen or tuple(n) in fill_coordinates: continue
          if img_mat[n[0]][n[1]] == from_colour:
            fill_coordinates.append(n)
            # seen[tuple(n)] = True
      
      # return img_mat

    def get_neighbouring_coordinates(position):
      """
      search for neighbouring coordinates
      """
      row = position[0]
      col = position[1]

      neighbours = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1)
      ]

      return [n for n in neighbours if 0 <= n[0] < len(current_image) and 0 <= n[1] < len(current_image[1])]

    # extract info
    centre_row = position[0]
    centre_col = position[1]

    img_mat = edge_image.tolist()
    for row in range(-radius, radius+1):
      for col in range(-radius, radius+1):
        if (row**2 + col**2 <= radius**2):
          if not (0 <= centre_row + row < len(current_image)): continue
          if not (0 <= centre_col + col < len(current_image[0])): continue
          # if current_image[row, col].tolist() != from_colour: continue
          fill_coordinates.append((centre_row + row, centre_col + col))

    current_mat = current_image.tolist()
    fill(current_mat, img_mat)

    return np.array(current_mat).astype('uint8')

  def remove_and_inpaint(self, folder, name, image_np, mask_np, inpaint_config):
    MASK = 'mask'
    SHOW = 'show'
    SAVE = 'save'
    NAIVE = 'NaiveLinearInpainter'
    INTERMEDIATE = 'IntermediateLinearInpainter'
    EXEMPLAR = 'ExemplarInpainter'

    # a place to store results
    inpaint_results = {}
    inpaint_results_ordered = []

    # handle mask
    current_config = inpaint_config[MASK]
    if current_config[SHOW] or current_config[SAVE]:
      inpaint_results[MASK] = mask_np
      inpaint_results_ordered.append(MASK)

    # handle NaiveLinearInpainter
    current_config = inpaint_config[NAIVE]
    if current_config[SHOW] or current_config[SAVE]:
      inpainted_np = NaiveLinearInpainter().remove_and_inpaint(folder, name, image_np, mask_np)
      inpaint_results[NAIVE] = inpainted_np
      inpaint_results_ordered.append(NAIVE)

    # handle IntermediateLinearInpainter
    current_config = inpaint_config[INTERMEDIATE]
    if current_config[SHOW] or current_config[SAVE]:
      inpainted_np = IntermediateLinearInpainter().remove_and_inpaint(folder, name, image_np, mask_np)
      inpaint_results[INTERMEDIATE] = inpainted_np
      inpaint_results_ordered.append(INTERMEDIATE)

    # handle Exemplar
    current_config = inpaint_config[EXEMPLAR]
    if current_config[SHOW] or current_config[SAVE]:
      inpainted_np = ExemplarInpainter().remove_and_inpaint(folder, name, image_np, mask_np)
      inpaint_results[EXEMPLAR] = inpainted_np
      inpaint_results_ordered.append(EXEMPLAR)
    
    # handle showing and saving
    for inpainter in inpaint_results_ordered:
      if inpaint_config[inpainter][SHOW]:
        plt.imshow(inpaint_results[inpainter])
        plt.show()

      if inpaint_config[inpainter][SAVE]:
        io.save(mask_np.tolist(), name + '_' + inpainter, folder, False)

  

class Inpainter:
  MASK_REMOVE = [255, 255, 255]

  def __init__(self):
    pass

  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    pass

  
  def get_neighbours(self, coordinate, nrows, ncols):
    """
    Get four neighbouring coordinates surrounding coordinate
    Remove points that are negative or outside the dimension nrows x ncols
    """
    row = coordinate[0]
    col = coordinate[1]

    neighbours = [
      (row - 1, col),
      (row + 1, col),
      (row, col - 1),
      (row, col + 1)
    ]

    return [n for n in neighbours if 0 <= n[0] < nrows and 0 <= n[1] < ncols]

class NaiveLinearInpainter(Inpainter):
  """
  Simple inpainter that extracts a single layer of pixels surrounding the masking area
  and use them as predictors in a linear regression model
  """

  def __init__(self):
    pass

  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    """
    Assumes dim(image_np) == dim(mask_np)
    """
    print("Starting NaiveLinearInpainter")

    # extract dimensions
    nrows = image_np.shape[0]
    ncols = image_np.shape[1]

    # convert image to python list
    inpainted_image = np.copy(image_np).astype('uint8').tolist()
    image = image_np.tolist()
    mask = mask_np.tolist()

    # store coordinates of pixels that are to be removed
    to_remove = set()
    for row in range(nrows):
      for col in range(ncols):
        if mask[row][col] == self.MASK_REMOVE: to_remove.add((row, col))

    # gather surrounding pixel coordinates
    surrounding_to_remove = set()
    for coordinate in to_remove:
      row = coordinate[0]
      col = coordinate[1]
      neighbours = self.get_neighbours(coordinate, nrows, ncols)
      for neighbour in neighbours:
        if neighbour in surrounding_to_remove or neighbour in to_remove: continue
        surrounding_to_remove.add(neighbour)

    # return empty image if there are no neighbours to learn from
    if len(surrounding_to_remove) == 0: return np.zeros(image_np.shape)

    # generate predictors and responses
    Xs = []
    rs = []
    gs = []
    bs = []
    for surrounding_coordinates in surrounding_to_remove:

      # extract info
      row = surrounding_coordinates[0]
      col = surrounding_coordinates[1]
      surrounding_colour = image[row][col]
      
      # split rgb
      r = surrounding_colour[0]
      g = surrounding_colour[1]
      b = surrounding_colour[2]

      # add to pred/resp lists
      Xs.append([row, col])
      rs.append(r)
      gs.append(g)
      bs.append(b)

    # generate three models for each value (colour)
    model_r = LinearRegressionModel(np.array(Xs), np.array(rs))
    model_g = LinearRegressionModel(np.array(Xs), np.array(gs))
    model_b = LinearRegressionModel(np.array(Xs), np.array(bs))

    # train each model
    model_r.train()
    model_g.train()
    model_b.train()

    # fill inpainted_image with predictions
    for inpaint_coordinates in to_remove:
      row = inpaint_coordinates[0]
      col = inpaint_coordinates[1]

      # predict rgb
      r = int(model_r.predict(inpaint_coordinates))
      g = int(model_g.predict(inpaint_coordinates))
      b = int(model_b.predict(inpaint_coordinates))

      # combine rgb and add to inpainted image with min/max checking
      combined_rgb = [r, g, b]
      for i in range(len(combined_rgb)):
        colour = combined_rgb[i]
        if colour < 0: colour = 0
        elif colour > 255: colour = 255
        combined_rgb[i] = colour
      inpainted_image[row][col] = combined_rgb
    
    return np.array(inpainted_image).astype('uint8')


class IntermediateLinearInpainter(Inpainter):
  """

  """

  def __init__(self):
    pass

  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    """
    Assumes dim(image_np) == dim(mask_np)
    """
    print("Starting IntermediateLinearInpainter")

    # extract dimensions
    nrows = image_np.shape[0]
    ncols = image_np.shape[1]

    # convert image to python list
    inpainted_image = np.copy(image_np).astype('uint8').tolist()
    image = image_np.tolist()
    mask = mask_np.tolist()

    # store coordinates of pixels that are to be removed
    to_remove = set()
    # to_remove_rows = set()
    # to_remove_cols = set()
    for row in range(nrows):
      for col in range(ncols):
        if mask[row][col] == self.MASK_REMOVE:
          to_remove.add((row, col))
          # to_remove_rows.add(row)
          # to_remove_cols.add(cols)

    # gather surrounding pixel coordinates
    # surrounding_to_remove = set()
    local_regression_dict = {}

    # row and col regressors
    row_regressors = {}
    col_regressors = {}

    undetermined = set()

    for coordinate in to_remove:
      row = coordinate[0]
      col = coordinate[1]

      # find minrow and maxrow used for regression:
      minrow = row - 1
      while minrow >= 0:
        if mask[minrow][col] != self.MASK_REMOVE: break
        minrow -= 1

      maxrow = row + 1
      while maxrow < nrows:
        if mask[maxrow][col] != self.MASK_REMOVE: break
        maxrow += 1

      # find mincol and maxcol used for regression:
      mincol = col - 1
      while mincol >= 0:
        if mask[row][mincol] != self.MASK_REMOVE: break
        mincol -= 1

      maxcol = col + 1
      while maxcol < ncols:
        if mask[row][maxcol] != self.MASK_REMOVE: break
        maxcol += 1

      row_pair = (minrow, maxrow)
      col_pair = (mincol, maxcol)

      if row_pair == (-1, 256) and col_pair == (-1, 256):
        undetermined.add((row, col))
        continue

      local_regression_dict[coordinate] = {
        'row' : row_pair,
        'col' : col_pair
      }

      # regression

      row_models = None
      col_models = None

      # row regression, fix row
      if col_pair != (-1, ncols):
        Xs = []
        channels = [[], [], []]

        for c in col_pair:
          if -1 < c < ncols:
            Xs.append([row, c])
            for channel in range(3):
              channels[channel].append(image[row][c][channel])

        col_models = {
          'r' : LinearRegressionModel(np.array(Xs), np.array(channels[0])),
          'g' : LinearRegressionModel(np.array(Xs), np.array(channels[1])),
          'b' : LinearRegressionModel(np.array(Xs), np.array(channels[2]))
        }

        for m in col_models: col_models[m].train()

      # col regression, fix col
      if row_pair != (-1, nrows):
        Xs = []
        channels = [[], [], []]

        for r in row_pair:
          if -1 < r < nrows:
            Xs.append([r, col])
            for channel in range(3):
              channels[channel].append(image[r][col][channel])

        row_models = {
          'r' : LinearRegressionModel(np.array(Xs), np.array(channels[0])),
          'g' : LinearRegressionModel(np.array(Xs), np.array(channels[1])),
          'b' : LinearRegressionModel(np.array(Xs), np.array(channels[2]))
        }

        for m in row_models: row_models[m].train()

      # regress with available models
      models_used = 0
      models = [row_models, col_models]
      channel_initials = ['r', 'g', 'b']
      predictions = [0, 0, 0]

      for model in models:
        if model:
          for i in range(len(predictions)):
            predictions[i] += model[channel_initials[i]].predict([row, col])
          models_used += 1

      for i in range(len(predictions)):
        pred_i = predictions[i]
        pred_i = int(pred_i / models_used)
        if pred_i < 0: pred_i = 0
        elif pred_i > 255: pred_i = 255
        predictions[i] = pred_i

      inpainted_image[row][col] = predictions

    inpainted_np = np.array(inpainted_image).astype('uint8')
    if len(undetermined) > 0:
      new_mask = np.zeros(mask_np.shape)
      for undetermined_coordinates in undetermined:
        new_mask[undetermined_coordinates[0]][undetermined_coordinates[1]] = np.array([255, 255, 255])
      return IntermediateLinearInpainter().remove_and_inpaint(folder, name, inpainted_np, new_mask)

    return inpainted_np


class ExemplarInpainter(Inpainter):
  def __init__(self):
    pass
  
  def get_neighbours(self, coordinates, img):
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


  def get_squared_difference(self, arr1, arr2):
    """
    assumes len(arr1) == len(arr2)
    """
    to_return = 0
    for i in range(len(arr1)):
      to_return += pow((arr1[i] - arr2[i]), 2)
    return to_return


  def normalised(self, nparr):
    norm = np.linalg.norm(nparr)
    max_val = np.amax(nparr)
    return nparr / max_val

  def get_patch_points(self, patch, mask, edges, num_patch_elements):
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
      if mask[coordinate[0]][coordinate[1]] == self.MASK_NONE: points_size += 1
    points_edges /= num_patch_elements
    points_size
    return (points_size, points_edges)

  def get_patch_masked_num(self, patch, mask):
    """
    Returns number of unmasked pixels
    """
    unmasked = 0
    for coordinate in patch:
      if mask[coordinate[0]][coordinate[1]] != self.MASK_NONE: unmasked += 1
    return unmasked

  def get_important_patches(self, to_fill, mask, edges, num_patch_elements, max_patch_size = None):
    """
    traverses to_fill (points that needs to be inpainted)
    """
    to_return = []
    operation_count = 0

    # min and max masked elements in a patch arbitrarily chosen
    min_patch_size = int(0.3 * num_patch_elements)
    if not max_patch_size:
      max_patch_size = int(0.7*num_patch_elements)
    # max_patch_size = num_patch_elements

    for fill_coordinates in to_fill:
      # optimisation: skip non-border points
      neighbours = [n for n in self.get_neighbours(fill_coordinates, mask) if mask[n[0]][n[1]] == self.MASK_NONE]
      if len(neighbours) == 0: continue
      
      # extract patches this coordinate appears in
      patches_appear_in = self.coordinates_to_patch[fill_coordinates]

      # consider each of these patches and measure each patch by importance
      viable_patches = []
      for patch_id in patches_appear_in:
        patch = self.patch_to_coordinates[patch_id]
        patch_information = self.get_patch_points(patch, mask, edges, num_patch_elements)
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

    if len(to_return) == 0: 
      return self.get_important_patches(to_fill, mask, edges, num_patch_elements, max_patch_size + 2)
    return to_return

  def get_similar_patch(self, identified_patch_id, img, mask, edges, MAX_SEARCH_RADIUS = 16):
    """


    """

    MIN_SEARCH_RADIUS = 0

    identified_patch = self.patch_to_coordinates[identified_patch_id]
    identified_topleft = identified_patch[0]

    ncoordinates = len(identified_patch)  # extract the length of elements in identified_patch
    candidate_patches = []
    for candidate_patch_id in self.full_patches:
      # if candidate_patch_id == 
      candidate_patch = self.patch_to_coordinates[candidate_patch_id]
      candidate_topleft = candidate_patch[0]

      distance_from_identified = math.sqrt(self.get_squared_difference(identified_topleft, candidate_topleft))
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
        if mask[identified_row][identified_col] != self.MASK_NONE:
          continue
        
        points_rgb += self.get_squared_difference(img[identified_row][identified_col], img[candidate_row][candidate_col])
        available_information += 1

      points = points_rgb
      heapq.heappush(candidate_patches, (points, available_information, candidate_patch_id))
    
    if len(candidate_patches) == 0: return self.get_similar_patch(identified_patch_id, img, mask, edges, MAX_SEARCH_RADIUS * 2)
    
    most_similar_tup = heapq.heappop(candidate_patches)
    most_similar = most_similar_tup[-1]
    return most_similar
    
  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    """

    """
    print("Starting ExemplarInpainter")
    img = image_np.tolist()

    # build edges
    edges = CannyEdgeDetection().generate_edges(image_np).tolist()

    # build mask
    self.MASK_NONE = [0, 0, 0]
    mask_np[mask_np > 126] = 255
    mask_np[mask_np <= 126] = 0
    mask = mask_np.tolist()

    # patch information
    PATCH_SIZE = 7
    num_patch_elements = PATCH_SIZE * PATCH_SIZE

    # contained information
    self.patch_to_coordinates = {}  # maps a patch to coordinates that live inside it
    self.coordinates_to_patch = {}  # maps a coordinate to all patches it exists in

    # identifier mappings
    self.identifier_patch_to_coordinates = {}  # maps a patch_id to its top left-corner coordinate
    self.identifier_coordinates_to_patch = {}  # maps a coordinate to the patch where it is the most top left coordinate (if exists)

    # sets
    to_fill = set()  # set containing coordinates that require filling
    self.full_patches = set()  # set containing ids of patches that are completely filled
    self.partial_patches = set()  # set containing ids of patches that are not completely filled

    patch_id = 0

    # build patches 
    # iterate over array
    nrows = len(img)
    ncols = len(img[0])
    for row in range(nrows - PATCH_SIZE):
      for col in range(ncols - PATCH_SIZE):
        self.patch_to_coordinates[patch_id] = []
        self.full_patches.add(patch_id)  # remove partial patches after

        # do identifier mappings
        self.identifier_coordinates_to_patch[(row, col)] = patch_id
        self.identifier_patch_to_coordinates[patch_id] = (row, col)

        # iterate over PATCH_SIZE x PATCH_SIZE window
        for prow in range(PATCH_SIZE):
          for pcol in range(PATCH_SIZE):

            # sum (trow = total row, tcol = total col)
            trow = row + prow
            tcol = col + pcol
            coordinates = (trow, tcol)

            self.patch_to_coordinates[patch_id].append(coordinates)

            # add coordinates to to_fill if it doesn't have a black mask pixel
            # i.e. this pixel is required to be inpainted
            if mask[trow][tcol] != self.MASK_NONE:
              to_fill.add(coordinates)
              self.partial_patches.add(patch_id)
            
            # build patch information into dicts
            if coordinates not in self.coordinates_to_patch: self.coordinates_to_patch[coordinates] = []
            self.coordinates_to_patch[coordinates].append(patch_id)

        # increment patch
        patch_id += 1

    # remove partial patches
    for patch_id in self.partial_patches:
      self.full_patches.remove(patch_id)

    # choosing the importance of which point to fill
    # format -> (no. surrounding unmasked, edge point, heap_operation_count, ..)
    # extract importance of patches
    importance = self.get_important_patches(to_fill, mask, edges, num_patch_elements)

    # begin inpatinging
    steps = 0
    while to_fill:
      
      # re-calibrate important points when required
      if len(importance) == 0:
        importance = self.get_important_patches(to_fill, mask, edges, num_patch_elements)
      
      # extract patch to be inpainted
      important_tuple = heapq.heappop(importance)
      num_unmasked = important_tuple[1]

      # this is the patch we will use
      identified_patch_id = important_tuple[2]
      identified_patch = self.patch_to_coordinates[identified_patch_id]

      # early exit for repeated work
      if identified_patch_id in self.full_patches:
        continue

      # find the most similar patch to identified_patch
      similar_patch_id = self.get_similar_patch(identified_patch_id, img, mask, edges)
      similar_patch = self.patch_to_coordinates[similar_patch_id]

      # fill coordinates
      for index in range(len(identified_patch)):
        identified_coordinate = identified_patch[index]
        identified_row = identified_coordinate[0]
        identified_col = identified_coordinate[1]

        similar_coordinate = similar_patch[index]
        similar_row = similar_coordinate[0]
        similar_col = similar_coordinate[1]

        # fill img if it is a point we are trying to fill
        if mask[identified_row][identified_col] != self.MASK_NONE:
          img[identified_row][identified_col] = img[similar_row][similar_col]

      # update full and partial patches
      self.partial_patches.remove(identified_patch_id)
      self.full_patches.add(identified_patch_id)

      # remove coordinates that have been filled and update our mask
      to_remove = set()
      coordinates_to_fill = {c for c in identified_patch if mask[c[0]][c[1]] != self.MASK_NONE}
      for coordinates in coordinates_to_fill:
        to_fill.remove(coordinates)
        mask[coordinates[0]][coordinates[1]] = self.MASK_NONE

        # connected patches to filled coordinates
        for connected_patch_id in self.coordinates_to_patch[coordinates]:
          connected_patch = self.patch_to_coordinates[connected_patch_id]

          # if the connected patch is now filled
          if connected_patch_id in self.full_patches: continue
          if self.get_patch_masked_num(connected_patch, mask) == 0:
            if connected_patch_id in self.partial_patches:
              self.partial_patches.remove(connected_patch_id)
              self.full_patches.add(connected_patch_id)

      steps += 1
    return np.array(img)
