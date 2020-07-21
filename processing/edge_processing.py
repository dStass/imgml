class EdgeDetection:
  def EdgeDetection(self):
    pass
  
  def detect_edges(self, img_mat, k_radius=2):
    ROW = len(img_mat)
    COL = len(img_mat[0])
    to_return = [[0 for p in range(ROW)] for q in range(COL)]

    for row in range(ROW):
      for col in range(COL):
        radius, total_count = self.get_k_radius(img_mat, (row, col), k_radius)

    return to_return

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
  
    return to_return, non_nulls


