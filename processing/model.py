import numpy as np
from sklearn.linear_model import LinearRegression

# full model
class Model:
  COL_MIN = 0
  COL_MAX = 255

  def __init__(self, split_factor = 1, image_matrix = None):

    # extract information from image
    self.image_matrix = image_matrix
    self.HEIGHT = len(image_matrix)
    self.WIDTH = len(image_matrix[0])

    # work out how many sub-models we will be using
    self.split_factor = split_factor
    self.split_by = pow(2, self.split_factor - 1)

    # generate the intervals of each submodel in the x and y directions
    self.height_interval = self.HEIGHT // self.split_by
    self.width_interval = self.WIDTH // self.split_by

    # generate training sub-models
    # assumes image is a square with sides a power of 2 and -ge pow(2, split_factor)
    sub_models = {}
    for row in range(0, self.HEIGHT, self.height_interval):
      for col in range(0, self.WIDTH, self.width_interval):

        # for each sub-model, generate:       
        # coordinates as predictors
        Xs = []

        # amounts of RED, GREEN, BLUE
        rs = []
        gs = []
        bs = []

        # extract data
        for r in range(row, min(row + self.height_interval, self.HEIGHT)):
          for c in range(col, min(col + self.width_interval, self.WIDTH)):
            Xs.append([r,c])
            rs.append(image_matrix[r][c][0])
            gs.append(image_matrix[r][c][1])
            bs.append(image_matrix[r][c][2])

        # generate three models for each value (colour)
        model_r = LinearRegressionModel(np.array(Xs), np.array(rs))
        model_g = LinearRegressionModel(np.array(Xs), np.array(gs))
        model_b = LinearRegressionModel(np.array(Xs), np.array(bs))

        # save sub_models
        sub_models[(row, col)] = [model_r, model_g, model_b]
    self.sub_models = sub_models

  def train(self):
    for sub_model_key in self.sub_models:
      sub_model_tuple = self.sub_models[sub_model_key]
      for sub_model in sub_model_tuple:
        sub_model.train()

  def predict(self, coordinates):

    # snap row and col 
    row, col = coordinates[0], coordinates[1]
    row = int(round(row - row % self.height_interval))
    col = int(round(col - col % self.width_interval))

    # identify the correct sub-model
    sub_model_tuple = self.sub_models[(row, col)]

    # make rgb prediction
    predicted = []
    for sub_model in sub_model_tuple:
      np_prediction = sub_model.predict(coordinates)
      prediction = np_prediction[0]
      if prediction < self.COL_MIN: prediction = self.COL_MIN
      elif prediction > self.COL_MAX: prediction = self.COL_MAX
      predicted.append(int(round(prediction)))

    return tuple(predicted)

class TrainingModel:
  def __init__(self, predictors, responses):
    self.predictors = predictors
    self.responses = responses

  def train(self):
    pass

  def predict(self, values):
    pass

class LinearRegressionModel(TrainingModel):
  def train(self):
    self.sk_regression = LinearRegression().fit(self.predictors, self.responses)

  def predict(self, predictors):
    return self.sk_regression.predict(np.reshape(np.array(predictors), (1,2) ))

class LogisticRegression(TrainingModel):
  def train(self):
    self.sk_regression = LinearRegression().fit(self.predictors, self.responses)

  def predict(self, predictors):
    return self.sk_regression.predict(np.reshape(np.array(predictors), (1,2) ))
