from processing.image_processing import ImageReader, ImageSaver
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



path = 'assets/shaggy.jpeg'
ir = ImageReader()

split_factor = 2
img_mat0 = ir.load(path)
img_mat = ir.load_quantised(path, 8)
ImageSaver().save(img_mat, "new_image2")


# e = EdgeDetection()
# # img_mat = [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]]
# # rcpair = [4,4]
# # k = 2
# # rad = e.get_k_radius(img_mat, rcpair, k)
# # print(rad)

detected = e.detect_edges(img_mat)
print(detected)