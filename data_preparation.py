import os
import random
import pandas

def seed():
    return 0.1


def randomize_data():
    if os.path.exists("randomized_data"):
        print("Randomized data already exists.")
    else:
        os.mkdir("randomized_data")

    for filename in os.listdir("training_data"):
        print("Randomizing " + filename)
        file = open("training_data/" + filename, "r")
        training_data = file.readlines()
        header = training_data[0]
        data = training_data[1:]
        random.shuffle(data, seed)
        data.insert(0, header)
        with open("randomized_data/" + filename[0:len(filename) - 4] + "_randomized.csv", "w") as target:
            for line in data:
                target.write(line)


def addClassifier(pixel):
    file = pandas.read_csv("top_pixels/top"+str(pixel)+"pixels_weka.csv")
    labels = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
    labels = labels.rename(columns={"0": "Class"})
    limit = pixel * 10
    if len(file.columns) < limit+1:
        file = pandas.concat([file, labels], axis=1)
    print(file)
    file.to_csv("top_pixels/top" + str(pixel) + "pixels_weka.csv",  index=False)


