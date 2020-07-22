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


# path = 'assets/colourful.jpg'
path = 'assets/el.png'
io = ImageIO(path.split('.')[1])
e = EdgeDetection()

split_factor = 2
# img_mat0 = ir.load(path)
# img_mat = io.load_quantised(path, 8)
img_mat = io.load_recursive_quantised(path, 24, 6)
# img_mat = e.apply_even_smoothing(img_mat, 1)
# img_mat = e.keep_bright(img_mat, 0.5)

io.save(img_mat, "new_image")

# # img_mat = [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]]
# # rcpair = [4,4]
# # k = 2
# # rad = e.get_k_radius(img_mat, rcpair, k)
# # print(rad)

# detected = e.detect_edges(img_mat, 2)
img_mat = e.generate_binary_edges_heatmap(img_mat, 2, 1)
repeats = 5
for i in range(repeats):
  img_mat = e.apply_even_smoothing(img_mat, 1)
  img_mat = e.keep_bright(img_mat, 0.5)
# img_mat = e.generate_binary_edges_heatmap(img_mat, 1, 3)
# img_mat = e.apply_even_smoothing(img_mat, 1)
# img_mat = e.keep_bright(img_mat, 0.8)
# img_mat = e.generate_binary_edges_heatmap(img_mat, 2)
# img_mat = e.generate_binary_edges_heatmap(img_mat, 2)
# img_mat = e.generate_binary_edges_heatmap(img_mat, 2)

# img_mat = e.generate_binary_edges_heatmap(img_mat, 2)

io.save(img_mat, "entropy")

# m = Model(8, entropy_heat_map)
# m.train()
# trained_img_mat = io.empty_matrix(len(img_mat), len(img_mat[0]))
# for row in range(len(entropy_heat_map)):
#   for col in range(len(entropy_heat_map[0])):
#     prediction = m.predict(([row, col]))
#     if (prediction != (0,0,0)): prediction = (255,255,255)
#     trained_img_mat[row][col] = prediction

# io.save(trained_img_mat, "regressed")
