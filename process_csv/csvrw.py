import csv
import random

class CSVReadWrite:
  def csv_to_list(self, data_path, shuffle=False, header=False):
    data = []
    with open(data_path, newline='') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',')
      for row in csvreader:
        data.append(row)
    if shuffle: random.shuffle(data)
    if header: data = data[1:]
    return data

  def list_to_csv(self, save_data, save_path):
    with open(save_path, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in save_data:
        writer.writerow(row)