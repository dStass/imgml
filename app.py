import math
import os

import numpy as np

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection, EntropyEdgeDetection
from processing.model import Model


# # # # # # # # # # #
#   START OF FILE   #
# # # # # # # # # # #


# declare paths
path = 'assets/'
outpath = 'output/'
filename = 'spaceman.jpg'
filepath = path + filename
filename_split = filepath.split('.')
name = filename_split[0]
ext = filename_split[1]

# reclare outpath
outpath = outpath + name + '/'
entropypath = outpath + 'entropy/'

# handle directories
# main directory
if not os.path.exists(outpath):
  os.makedirs(outpath)

# handle entropy directory
if not os.path.exists(entropypath):
  os.makedirs(entropypath)

# declare objects
io = ImageIO(ext)
e = EntropyEdgeDetection()


START_BINS = 24
END_BINS = 6
STEP_BINS = 6

PREV_BINS = START_BINS

# save entropy layers
entropy_mats = {}
for end_bins in range(START_BINS, END_BINS - 1, -STEP_BINS):
  # load and resize original image
  img_mat = io.load_recursive_quantised(filepath, PREV_BINS, END_BINS, True)
  io.save(img_mat, "loaded_img" + str(PREV_BINS) + '_' + str(end_bins), path=outpath)
  PREV_BINS -= STEP_BINS

  for kradius in range(1,4):
    entropy_map = e.detect_edges(img_mat, kradius)
    entropy_mat = e.generate_binary_edges_heatmap(img_mat, entropy_map, kradius, 0)

    mat_key = 'bins' + str(end_bins) + '_kradius' + str(kradius)
    entropy_mats[mat_key] = entropy_mat

    # io.save(entropy_mat, "kradius" + str(kradius) + "_bins" + str(end_bins), path=entropypath)

for key in entropy_mats:
  entropy_mat = entropy_mats[key]
  io.save(entropy_mat, str(key), path=entropypath)

print(entropy_mats)
final_mat = e.combine_img_mats(entropy_mats)
io.save(final_mat, "FINAL", path=outpath)

# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #