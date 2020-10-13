import random, os

def seed():
    return 0.31415926535898

def randomize_data():
    if os.path.exists("randomized_data"):
        print("Randomized data already exists.")
    else:
        os.mkdir("randomized_data")
        for filename in os.listdir("training_data"):
            print("Randomizing " + filename)
            file = open("training_data/" + filename, "r")
            training_data = file.readlines()
            random.shuffle(training_data, seed)
            with open("randomized_data/" + filename[0:len(filename) - 4] + "_randomized.csv", "w") as target:
                for line in training_data:
                    target.write(line)