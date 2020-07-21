from processing.image_processing import ImageReader, ImageSaver
from processing.model import Model

path = 'assets/shaggy.jpeg'
ir = ImageReader()

split_factor = 4
img_mat = ir.load(path)

print("Loaded")
m = Model(split_factor, img_mat)
print("Model created")
m.train()
print("Model trained")
trained_img_mat = ImageReader().empty_matrix(len(img_mat), len(img_mat[0]))
for row in range(len(img_mat)):
  for col in range(len(img_mat[0])):
    prediction = m.predict(([row, col]))
    trained_img_mat[row][col] = prediction
print("New Image created")
ImageSaver().save(trained_img_mat, "new_colourful")
print("Task completed")