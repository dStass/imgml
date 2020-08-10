# # # # # # # # # # #
#   START OF FILE   #
# # # # # # # # # # #

import math
import json
import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection, EntropyEdgeDetection, CannyEdgeDetection
from processing.model import Model
from processing.inpaint_processing import ImageInpainter
from process_csv.csvrw import CSVReadWrite

def generate_mask(img_np, radius):
  """
  randomly generates a mask of a line
  """
  shape = img_np.shape
  nrows = shape[0]
  ncols = shape[1]
  to_return = np.zeros(shape).tolist()
  centre0 = [0,0]

  # randomly find a centre for the circle
  while centre0[0] - radius < 0 or centre0[0] + radius >= nrows or centre0[1] - radius < 0 or centre0[1] + radius >= ncols:
    rand_row = int(random.random() * nrows)
    rand_col = int(random.random() * ncols)
    centre0 = [rand_row, rand_col]

  centre1 = [0,0]
  # randomly find a second centre
  while centre1[0] - radius < 0 or centre1[0] + radius >= nrows or centre1[1] - radius < 0 or centre1[1] + radius >= ncols or centre0[1] == centre1[1]:
    rand_row = int(random.random() * nrows)
    rand_col = int(random.random() * ncols)
    centre1 = [rand_row, rand_col]

  if centre0[1] < centre1[1]:
    centre0, centre1 = centre1, centre0
  
  gradient = (centre0[0] - centre1[0]) / (centre0[1] - centre1[1])

  # y = mx + b -> b = y - mx
  intercept = centre0[0] - gradient * centre0[1]

  centres = []
  for i in range(centre1[1], centre0[1] + 1):
    centres.append([int(gradient * i + intercept), i])

  for centre in centres:
    for row in range(centre[0] - radius, centre[0] + radius + 1):
      for col in range(centre[1] - radius, centre[1] + radius + 1):
        if math.sqrt(pow(row - centre[0], 2) + pow(col - centre[1], 2)) <= radius:
          to_return[row][col] = [255, 255, 255]

  return np.array(to_return).astype('uint8')



# start of program

config = {}
with open('config.json') as json_file:
    config = json.load(json_file)

# program mode
PROGRAM_MODE = config['PROGRAM_MODE']
PROGRAM_MODE_0 = 0
PROGRAM_MODE_1 = 1

# Export settings
OUTPUT_FOLDER = config['OUTPUT_FOLDER']
OUTPUT_NAME = config['OUTPUT_NAME']

# import image
IMG_PATH = config['IMG_PATH']
ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)
loaded_image = io.load(IMG_PATH)

# analysis related settings
analysis_config = config['analysis']
TRIALS_PER_TEST_RADIUS = analysis_config['TRIALS_PER_TEST_RADIUS']
# radius_details = analysis_config['radius']

inpaint_config = config['inpaint']

# declare inpainter
ip = ImageInpainter()

# reporting information
full_report = {
  'NaiveLinearInpainter' : [],
  'IntermediateLinearInpainter' : [],
  'ExemplarInpainter' : []
}


# for radius in range(radius_details['start'], radius_details['end'], radius_details['step']):
# csv_list = [[k for k in full_report]]
radius = analysis_config['radius']
for trial in range(TRIALS_PER_TEST_RADIUS):
  mask = generate_mask(loaded_image, radius)
  returned_report = ip.remove_and_inpaint(OUTPUT_FOLDER, OUTPUT_NAME, loaded_image, mask, inpaint_config, loaded_image)
  # new_entry = []
  for each_report in returned_report:
    # if radius not in full_report[each_report]:
    #   full_report[each_report][radius] = []
    full_report[each_report].append(returned_report[each_report])
    # new_entry.append(returned_report[each_report])
  # csv_list.append(new_entry)
#   print(returned_report)

# print(full_report)

csv_list = []
for key in full_report:
  new_list = [key] + [str(val) for val in full_report[key]]
  csv_list.append(new_list)

csvrw = CSVReadWrite()
csvrw.list_to_csv(csv_list, OUTPUT_FOLDER+OUTPUT_NAME+'_REPORT')



# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #