from PIL import Image

class ImageReader:
  def __init__(self):
    pass

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

  def empty_matrix(self, height, width):
    return [[0 for w in range(width)] for h in range(height)]

class ImageSaver:
  def __init__(self):
    pass

  def save(self, img_mat, name, path=''):
    ROW, COL = len(img_mat), len(img_mat[0])
    to_save = Image.new('RGB', (ROW, COL))
    to_save_loaded = to_save.load()
    for row in range(ROW):
      for col in range(COL):
        to_save_loaded[row, col] = img_mat[row][col]
    to_save.save(path + name + ".jpg", "JPEG", optimize=True) 