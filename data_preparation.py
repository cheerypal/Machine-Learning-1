import random, os
def randomize_data():
    if os.path.exists("randomized_data"):
        print("Randomized training training_data already exists.")
    else:
        os.mkdir("randomized_data")
        for filename in os.listdir("training_data"):
            print("Randomizing " + filename)
            with open("training_data/" + filename, "r") as source:
                data = [(random.random(), line) for line in source]
            data.sort()
            with open("randomized_data/" + filename[0:len(filename)-4] + "_randomized.csv", "w") as target:
                for _, line in data:
                    target.write(line)