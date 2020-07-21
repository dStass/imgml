from PIL import Image
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.cluster import KMeans


class ImageIO:
  IMG_SIZE = 512

  def __init__(self, ext='jpg'):
    self.extension = ext
    self.FORMAT = 'JPEG'
    if ext == 'png':
      self.FORMAT = 'PNG'

  def load(self, path):
    im = Image.open(path, 'r')
    height, width = im.size
    im_loaded = im.load()

    to_return = self.empty_matrix(height, width)
    for row in range(width):
      for col in range(height):
        colour = im_loaded[row, col]
        to_return[row][col] = colour
    return to_return

  def load_quantised(self, path, num_colours = -1):
    """
      load an image but reduce colours into a given number of colour bins
    """
    im = io.imread(path)
    im = resize(im, (self.IMG_SIZE, self.IMG_SIZE), anti_aliasing=True)
    im = self.IMG_SIZE * im
    if im.shape[2] == 4:
      im = np.delete(im, [3], axis=2)
    arr = im.reshape((-1,3))
    kmeans = KMeans(n_clusters=num_colours, random_state=42).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    reduced_img = centers[labels].reshape(im.shape).astype('uint8')
    reduced_transpose = np.transpose(reduced_img, [1,0,2])
    return [[tuple(r) for r in l] for l in reduced_transpose.tolist()]

  def empty_matrix(self, height, width):
    return [[0 for w in range(width)] for h in range(height)]

  def save(self, img_mat, name, path=''):
    ROW, COL = len(img_mat), len(img_mat[0])
    to_save = Image.new('RGB', (ROW, COL))
    to_save_loaded = to_save.load()
    for row in range(ROW):
      for col in range(COL):
        to_save_loaded[row, col] = img_mat[row][col]
    to_save.save(path + name + '.' + self.extension, self.FORMAT, optimize=True)
