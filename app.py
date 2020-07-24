import math
import os

import numpy as np

from processing.image_processing import ImageIO
from processing.edge_processing import EdgeDetection
from processing.model import Model

# path = 'assets/shaggy.jpeg'
# ir = ImageReader()

# split_factor = 2
# img_mat = ir.load(path)

# print("Loaded")
# m = Model(split_factor, img_mat)
# print("Model created")
# m.train()
# print("Model trained")
# trained_img_mat = ImageReader().empty_matrix(len(img_mat), len(img_mat[0]))
# for row in range(len(img_mat)):
#   for col in range(len(img_mat[0])):
#     prediction = m.predict(([row, col]))
#     trained_img_mat[row][col] = prediction
# print("New Image created")
# ImageSaver().save(trained_img_mat, "new_colourful")
# print("Task completed")

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
e = EdgeDetection()

# load and resize original image
img_mat = io.load_recursive_quantised(filepath, 24, 12)
# entropy_map = e.detect_edges()

# save entropy layers
entropy_mats = {}
entropy_maps = {}
for kradius in range(1,4):
  # for cutoff in np.arange(-4, 4.1, 0.5):
  if kradius not in entropy_maps:
    entropy_maps[kradius] = e.detect_edges(img_mat, kradius)
  entropy_map = entropy_maps[kradius]
  entropy_mat = e.generate_binary_edges_heatmap(img_mat, entropy_map, kradius, 0)
  entropy_mats[kradius] = entropy_mat
  io.save(entropy_mat, "kradius" + str(kradius) , path=entropypath)
  # entropy_layers.append(entropy_mat)

for key in entropy_mats:
  entropy_mat = entropy_mats[key]
  io.save(entropy_mat, "kradius" + str(key), path=entropypath)

img_mat = e.combine_img_mats(entropy_mats)

# # # # # # # # # # #
#    END OF FILE    #
# # # # # # # # # # #