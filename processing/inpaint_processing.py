import numpy as np

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