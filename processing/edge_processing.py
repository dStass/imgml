import math
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
import cv2


class EdgeDetection:
  def EdgeDetection(self):
    pass
  
  def colour_transform(self, entropy_value):
    r = int(abs(entropy_value) * 255)
    g = int(abs(entropy_value) * 255)
    b = int(abs(entropy_value) * 255)
    return (r, g, b)

  def get_k_radius(self, img_mat, rcpair, k):
    """
      
    """
    centre_row = rcpair[0]
    centre_col = rcpair[1]

    ROW = len(img_mat)
    COL = len(img_mat[0])

    to_return = [[None for p in range(-k, k + 1)] for q in range(-k, k + 1)]
    non_nulls = 0
    for r in range(-k, k + 1):
      for c in range(-k, k + 1):
        row = centre_row + r
        col = centre_col + c
        if not (0 <= row < ROW): continue
        if not (0 <= col < COL): continue
        to_return[r+k][c+k] = img_mat[row][col]
        non_nulls += 1
  
    return to_return


  def keep_bright(self, img_mat, brightness=0.5):
    ROW = len(img_mat)
    COL = len(img_mat[0])
    scores = [[0 for p in range(ROW)] for q in range(COL)]
    to_return = [[(0,0,0) for _ in r] for r in scores]

    for row in range(ROW):
      for col in range(COL):
        scores[row][col] = sum(img_mat[row][col]) / len(img_mat[row][col])
        if scores[row][col] >= brightness*255:
          to_return[row][col] = self.colour_transform(1)
    
    return to_return

  def apply_even_smoothing(self, img_mat, k_radius=2):
    ROW = len(img_mat)
    COL = len(img_mat[0])
    to_return = [[0 for p in range(ROW)] for q in range(COL)]

    for row in range(ROW):
      for col in range(COL):
        average_colour = [0,0,0]
        radius = self.get_k_radius(img_mat, (row, col), k_radius)

        # get colour sum
        total_colours = 0
        for radius_row in range(len(radius)):
          for radius_col in range(len(radius[0])):
            rad_rgb = radius[radius_row][radius_col]
            if rad_rgb:
              average_colour[0] += rad_rgb[0]
              average_colour[1] += rad_rgb[1]
              average_colour[2] += rad_rgb[2]
              total_colours += 1
        
        # get average
        # average_colour[0] = float(average_colour[0])/float(total_colours)
        # average_colour[1] = float(average_colour[1])/float(total_colours)
        # average_colour[2] = float(average_colour[2])/float(total_colours)
        if sum(average_colour) != 0:
          average_colour = [255,255,255]
        else:
          average_colour = [0,0,0]
        # update to_return
        to_return[row][col] = tuple(average_colour)
    return to_return


  def combine_img_mats(self, img_mats, ROWS = None, COLS = None):
    if not ROWS or not COLS:
      for key in img_mats:
        img_mat = img_mats[key]
        ROWS = len(img_mat)
        COLS = len(img_mat[0])
        break
    
    to_return = [[(0,0,0) for p in range(ROWS)] for q in range(COLS)]

    # and operation on all pixels 
    for row in range(ROWS):
      for col in range(COLS):
        pixel_sum = [0,0,0]
        for key in img_mats:
          img_mat = img_mats[key]
          rgb = img_mat[row][col]
          pixel_sum[0] += rgb[0]
          pixel_sum[1] += rgb[1]
          pixel_sum[2] += rgb[2]
  
        for i in range(len(pixel_sum)):
          pixel_sum[i] = int(pixel_sum[i] / len(img_mats))
        
        # keep = False
        if pixel_sum[0] >= 128 and pixel_sum[1] >= 128 and pixel_sum[2] >= 128:
          to_return[row][col] = self.colour_transform(1)
        
    return to_return

class EntropyEdgeDetection(EdgeDetection):
  def __init__(self):
    pass

  def generate_binary_edges_heatmap(self, im, entropy_map = None, k_radius=2, z_val=1):
    """


    """
    img_mat = [[tuple(c) for c in r] for r in im.tolist()]

    if not entropy_map:
      entropy_map = self.detect_edges(img_mat, k_radius)
    rows = len(entropy_map)
    cols = len(entropy_map[0])

    # calculate mean
    entropy_count = 0
    entropy_sum = 0
    for row in range(rows):
      for col in range(cols):
        if entropy_map[row][col] == 0: continue
        entropy_count += 1
        entropy_sum += entropy_map[row][col]
    entropy_mean = entropy_sum / entropy_count

    # calculate variance and sd
    var_running_sum = 0
    for row in range(rows):
      for col in range(cols):
        # if entropy_map[row][col] == 0: continue
        var_running_sum += math.pow(entropy_map[row][col] - entropy_mean, 2)
    variance = var_running_sum / (entropy_count - 1)
    sd = math.sqrt(variance)

    z_val = -1
    upper_one_sd = entropy_mean + z_val * (sd / math.sqrt(entropy_count))
    to_return = [[(0,0,0) for _ in r] for r in entropy_map]
    for row in range(rows):
      for col in range(cols):
        entropy = entropy_map[row][col]
        if entropy > upper_one_sd:
          # to_return[row][col] = img_mat[row][col]
          to_return[row][col] = self.colour_transform(1)
    
    return np.array(to_return).astype('uint8')

  def entropy_img(self, img_mat):
    """
    Returns the entropy of a rectangle by adding the pixel colours into buckets
    calculates the probability (count of elements divided by total) of a particular bucket
    """
    ROW = len(img_mat)
    COL = len(img_mat[0])
    
    buckets = {}
    entropy = 0
    total = 0
    for row in range(ROW):
      for col in range(COL):
        colour = img_mat[row][col]
        if not colour: continue
        if colour not in buckets: buckets[colour] = 0
        buckets[colour] += 1
        total += 1
    
    # calculate total entropy
    if len(buckets) == 1: return 0
    for b in buckets:
      count = buckets[b]
      p = count / total
      entropy += p * math.log(p)
    
    entropy = -entropy
    return(min(1.0, entropy))



  def detect_edges(self, img_mat, k_radius=2):
    """

    """
    ROW = len(img_mat)
    COL = len(img_mat[0])
    to_return = [[0 for p in range(ROW)] for q in range(COL)]

    for row in range(ROW):
      for col in range(COL):
        radius = self.get_k_radius(img_mat, (row, col), k_radius)
        entropy = self.entropy_img(radius)
        to_return[row][col] = entropy

    return to_return

class CannyEdgeDetection(EdgeDetection):
  def __init__(self):
    pass


  def transform_greyscale(self, rgb_tup):
    """
    using rgb transformation numbers from 
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html
    """

    r = rgb_tup[0]
    g = rgb_tup[1]
    b = rgb_tup[2]

    y = 0.2125*r + 0.7154*g + 0.0721*b

    y = int(y)

    return [y, y, y]

  def generate_edges(self, image):
    img_mat = [[self.transform_greyscale(c) for c in r] for r in image.tolist()]
    # img_mat = self.apply_even_smoothing(img_mat,2)
    # ROWS = len(img_mat)
    # COLS = len(img_mat[0])

    # # for row in range(ROWS):
    # #   for col in range(COLS):
    # #     if img_mat[row][col] != (0,0,0):
    # #       img_mat[row][col] == (255,255,255)

    # image = np.array(img_mat).astype('uint8')
    image = cv2.Canny(image, 245, 255)
    image = np.array([[[c, c, c] for c in r] for r in image.tolist()]).astype('uint8')
    # img_mat = [[(c*255, c*255, c*255) for c in r] for r in image.tolist()]
    # # for row in img_mat:
    # #   for col in 
    # img_mat = self.apply_even_smoothing(img_mat,1)

    return image
