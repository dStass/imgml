import time

import cv2
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

from processing.image_processing import ImageIO


def normalised(nparr):
  norm = np.linalg.norm(nparr)
  max_val = np.amax(nparr)
  return nparr / max_val

IMG_PATH = 'assets/car.jpg'

ext = IMG_PATH.split('.')[1]
io = ImageIO(ext)

# loaded_images = io.load_recursive_quantised(IMG_PATH, 20, 2, 19)
# img = loaded_images[0]

loaded_image = io.load(IMG_PATH)
# loaded_greyscale = (io.load_greyscale(IMG_PATH).astype('float64')/255)

# first we denoise
denoised_image = (denoise_tv_chambolle(loaded_image, weight=0.1, multichannel=True) * 255).astype('uint8')

# extract texture
texture_image = io.convert_greyscale(loaded_image) - io.convert_greyscale(denoised_image)
texture_image = normalised(texture_image - np.amin(texture_image))*255

# restored image
restored_lst = denoised_image.tolist()
texture_lst = texture_image.tolist()

nrows = len(texture_lst)
ncols = len(texture_lst[0])

for row in range(nrows):
  for col in range(ncols):
    intensity = texture_lst[row][col][0]/255
    inverse_intensity = 1 - intensity
    for i, channel in enumerate(restored_lst[row][col]):
      restored_lst[row][col][i] = channel * inverse_intensity

restored = np.array(restored_lst).astype('uint8')

plt.imshow(loaded_image)
plt.show()

plt.imshow(denoised_image)
plt.show()

plt.imshow(texture_image)
plt.show()

plt.imshow(restored)
plt.show()



# _T = 0.2
# _lambda = 0.001

# PATCH_SIZE = 7

# N = 200
# eps = 0.05

# t0 = time.time()

# channel_dict = {
#   '_I_0' : np.copy(loaded_greyscale),
#   '_I_n' : np.copy(loaded_greyscale)
# }



# for n in range(1, N+1):
#   _I_0 = channel_dict['_I_0']
#   _I_n = channel_dict['_I_n']

#   # part A of equation
#   _A = np.copy(_I_n)

#   # part B of equation
#   _B = _T * _lambda * (_I_0 - _I_n)

#   # part C of equation

#   # spatial gradients
#   # _I_x = 0.5 * (np.roll(_I_n, -1, axis=1) - np.roll(_I_n, 1, axis=1))
#   # _I_xx = np.roll(_I_n, -1, axis=1) - 2 * _I_n * np.roll(_I_n, 1, axis=1)
#   # _I_y = 0.5 * (np.roll(_I_n, -1, axis=0) - np.roll(_I_n, 1, axis=0))
#   # _I_yy = np.roll(_I_n, -1, axis=0) - 2 * _I_n * np.roll(_I_n, 1, axis=0)
#   # _I_xy = 0.5 * (np.roll(_I_x, -1, axis=0) - np.roll(_I_x, 1, axis=0))

#   _I_x = ndimage.sobel(_I_n,axis=0,mode='constant')
#   _I_xx = ndimage.sobel(_I_x,axis=0,mode='constant')

#   _I_y = ndimage.sobel(_I_n,axis=1,mode='constant')
#   _I_yy = ndimage.sobel(_I_y,axis=1,mode='constant')

#   _I_xy = ndimage.sobel(_I_x,axis=1,mode='constant')

  
#   # square of gradients
#   _I_x_sq = _I_x * _I_x
#   _I_y_sq = _I_y * _I_y

#   # split C into numerator part and denominator part
#   _C_numerator = _I_xx * _I_y_sq + _I_yy * _I_x_sq - 2 * _I_x * _I_y * _I_xy
#   _C_denominator = np.power(eps + _I_x_sq + _I_y_sq, 3/2)

#   # divide numerator by denominator and multiply by step size
#   _C = _T * _C_numerator / _C_denominator

#   # combine and normalise _I_n_plus_1 into 0-1 range
  
#   _I_n_plus_1 = _A + _B + _C
#   _I_n_plus_1 = _I_n_plus_1 - np.amin(_I_n_plus_1)
#   _I_n_plus_1 = normalised(_I_n_plus_1)
  
#   channel_dict['_I_n'] = _I_n_plus_1
#   print()

# _I_n = (channel_dict['_I_n'] * 255).astype('uint8')
# plt.imshow(_I_n)
# plt.show()

# # use intensities to denoise image


print()

# img = np.true_divide(img.astype('float64'), 255)
# R, G, B = cv2.split(img)

# R = R.astype('float64')
# G = G.astype('float64')
# B = B.astype('float64')

# R *= (1/255)
# G *= (1/255)
# B *= (1/255)

# eps = 0.00000000005

# channels = {
#   'B' : {
#     '_I_0' : np.copy(B),
#     '_I_n' : np.copy(B)
#   },
#   'G' : {
#     '_I_0' : np.copy(G),
#     '_I_n' : np.copy(G)
#   },
#   'R' : {
#     '_I_0' : np.copy(R),
#     '_I_n' : np.copy(R)
#   },

# }

# _I_0 = np.copy(g)
# _I_n = np.copy(_I_0)

# saved_plots = []
# channel_aggregate = []

# start iteration
# for n in range(1, N+1):

#   for channel in channels:

#     # unpack info
#     channel_dict = channels[channel]
#     _I_0 = channel_dict['_I_0']
#     _I_n = channel_dict['_I_n']

#     # part A of equation
#     _A = np.copy(_I_n)

#     # part B of equation
#     _B = _T * _lambda * (_I_0 - _I_n)

#     # part C of equation

#     # spatial gradients
#     _I_x = 0.5 * (np.roll(_I_n, -1, axis=1) - np.roll(_I_n, 1, axis=1))
#     _I_xx = np.roll(_I_n, -1, axis=1) - 2 * _I_n * np.roll(_I_n, 1, axis=1)
#     _I_y = 0.5 * (np.roll(_I_n, -1, axis=0) - np.roll(_I_n, 1, axis=0))
#     _I_yy = np.roll(_I_n, -1, axis=0) - 2 * _I_n * np.roll(_I_n, 1, axis=0)
#     _I_xy = 0.5 * (np.roll(_I_x, -1, axis=0) - np.roll(_I_x, 1, axis=0))


#     _I_x_sq = _I_x * _I_x
#     # _I_x_mat = np.matrix(_I_x_sq)
#     _I_y_sq = _I_y * _I_y

#     # plt.imshow(_I_x_sq)
#     # plt.show()

#     _C_numerator = _I_xx * _I_y_sq + _I_yy * _I_x_sq - 2 * _I_x * _I_y * _I_xy
#     _C_denominator = np.power(eps + _I_x_sq + _I_y_sq, 3/2)

#     _C = _T * _C_numerator / _C_denominator

#     # _C[np.isnan(_C)] = 0

#     _I_n_plus_1 = _A + _B + _C

#     _I_n = _I_n_plus_1
#     # _I_n[_I_n < 0] = 0
#     # if np.amin(_I_n) < 0:
#     #   _I_n -= np.amin(_I_n)
#     # # _I_n[_I_n > 1] = 1
#     # _I_n = normalised(_I_n) * 1

#     channel_dict['_I_n'] = _I_n

#     if n % 1 == 0:
#       if len(channel_aggregate) == 3:
#         # add 
#         _I_n_B = channels['B']['_I_n']
#         _I_n_G = channels['G']['_I_n']
#         _I_n_R = channels['R']['_I_n']

#         _I_n_R -= np.amin(_I_n_R) if np.amin(_I_n_R) < 0 else 0         
#         _I_n_G -= np.amin(_I_n_G) if np.amin(_I_n_G) < 0 else 0
#         _I_n_B -= np.amin(_I_n_B) if np.amin(_I_n_B) < 0 else 0

#         # _I_n_R[_I_n_R < 0] = 0
#         # _I_n_G[_I_n_G < 0] = 0        
#         # _I_n_B[_I_n_B < 0] = 0

#         _I_n_R = normalised(_I_n_R)        
#         _I_n_G = normalised(_I_n_G)
#         _I_n_B = normalised(_I_n_B)


#         # stack RGB
#         stacked_channels = np.dstack((_I_n_R * 255, _I_n_G * 255, _I_n_B * 255)).astype('uint8')

#         saved_plots.append(stacked_channels)

#         # reset
#         channel_aggregate = []

#       else:
#         channel_aggregate.append(_I_n)

#       # b, g, r = cv2.split(_I_n.astype(int))
#       # b2 = np.insert(b, 0, [0, 0])
#       # saved_plots.append(_I_n)

#       # plt.subplot(_I_n)
#     # plt.imshow(_I_n)
#     # plt.show()

# for saved_plot in saved_plots[::-1]:
# # for saved_plot in saved_plots:
#   plt.imshow(saved_plot)
#   plt.show()


#   img_mat = img.tolist()
#   saved_plot_mat = saved_plot.tolist()

#   for row in range(len(img_mat)):
#     for col in range(len(img_mat[0])):
#       for channel in range(3):
#         img_mat[row][col][channel] -= saved_plot_mat[row][col][channel]
#         img_mat[row][col][channel] = max(0, img_mat[row][col][channel])
  


#   difference = np.array(img_mat)
#   # print(ndifference))
#   plt.imshow(difference)
#   plt.show()

# # difference = img - saved_plots[0]

# print("Time taken = ", time.time() - t0)





# # Get x-gradient in "sx"
# sx = ndimage.sobel(img,axis=0,mode='constant')
# # Get y-gradient in "sy"
# sy = ndimage.sobel(img,axis=1,mode='constant')
# # plt.imshow(ndimage.sobel(sx,axis=0,mode='constant'))
# plt.imshow(sy)
# plt.show()