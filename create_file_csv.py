import os
import csv


path = "data/"
directories = os.listdir(path)
files = []

for directory in directories:
    all_files = os.listdir(path + directory)
    for video in all_files:
        video_path = path + directory + "/" + video
        files.append([video_path, directory])

with open("./data/videos.csv", "w+") as f:
     print(f)
     print(dir(f))
     writer = csv.writer(f)
     writer.writerows(files)

