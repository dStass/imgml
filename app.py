import math
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

def colour_kernel(entropy):
  r, g, b = int(abs(entropy) * 255), int(abs(entropy) * 255), int(abs(entropy) * 255)
  return (r, g, b)

def generate_heat_map(entropy_map):
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
      if entropy_map[row][col] == 0: continue
      var_running_sum += math.pow(entropy_map[row][col] - entropy_mean, 2)
  variance = var_running_sum / entropy_mean
  sd = math.sqrt(variance)

  z_val = 0
  upper_one_sd = entropy_mean + z_val * (sd / math.sqrt(entropy_count))
  to_return = [[(0,0,0) for _ in r] for r in entropy_map]
  for row in range(rows):
    for col in range(cols):
      entropy = entropy_map[row][col]
      if entropy > upper_one_sd:
        to_return[row][col] = colour_kernel(entropy)
  
  return to_return

  # [[colour_kernel(c) for c in r] for r in entropy_map]

# path = 'assets/colourful.jpg'
path = 'assets/room2.jpg'
io = ImageIO(path[-3:])

split_factor = 2
# img_mat0 = ir.load(path)
img_mat = io.load_quantised(path, 6)
io.save(img_mat, "new_image")


e = EdgeDetection()
# # img_mat = [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]]
# # rcpair = [4,4]
# # k = 2
# # rad = e.get_k_radius(img_mat, rcpair, k)
# # print(rad)

detected = e.detect_edges(img_mat, 2)
entropy_heat_map = generate_heat_map(detected)
io.save(entropy_heat_map, "entropy")

print(detected)

