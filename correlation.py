import pandas

# Read training data and associated labels
data = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")

for x in range(0,10):
    labels = pandas.read_csv("randomized_data/y_train_smpl_"+str(x)+"_randomized.csv")
    labels.rename(columns = {0:2304+x},inplace=True)
    print(labels)
    data = pandas.concat([data,labels],axis=1)
    print(data)

#print(data.corr())