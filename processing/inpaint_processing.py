import numpy as np
import matplotlib.pyplot as plt

from processing.model import LinearRegressionModel

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

  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    inpaint_results = {}
    inpaint_results['mask'] = mask_np
    
    naive_linear_np = NaiveLinearInpainter().remove_and_inpaint(folder, name, image_np, mask_np)
    inpaint_results['naive_linear_np'] = naive_linear_np

    plt.imshow(naive_linear_np)
    plt.show()
    pass

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
  def __init__(self):
    pass

  def remove_and_inpaint(self, folder, name, image_np, mask_np):
    """
    Assumes dim(image_np) == dim(mask_np)
    """
    print("linreg inpainting")

    # plt.imshow(image_np)
    # plt.show()

    # plt.imshow(mask_np)
    # plt.show()

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