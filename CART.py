import easygui as easygui
import pandas as pd
from sklearn.model_selection import train_test_split

#Import hazlenuts file and drop sample_id
#Split data into test and train arrays
def test_train_data(d):
    dataset = pd.read_csv("hazlenuts.csv").drop(columns=['sample_id'])
    train, test = train_test_split(dataset, test_size=0.33)

    testdata = []

    for row in test.values:
        testdata.append((list(row)))

    trainingdata = []

    for row in train.values:
        trainingdata.append((list(row)))
    return testdata, trainingdata

#find the number of unique rows
# #find out how many different types of hazlenuts we have
def uniquerows(rows):
    num_rows = {}
    for row in rows:
        #gets the last column 'variety'
        r = row[-1]

        # go through each tow and see if it already exists in run_rows
        #increments it as it goes through
        if r not in num_rows:
            num_rows[r] = 0
        num_rows[r] = num_rows[r] + 1
    return num_rows


class Threshold:

    def __init__(self, col, val):
        self.col = col
        self.val = val

    def compare(self, e):
        sample_row = e[self.col]

        if sample_row >= self.val:
            return True

#get the training data and split it depending on threshholds
def splitrows(rows, threshold):
    row1, row2 = [], []

    for i in rows:
        if threshold.compare(i):
            row1.append(i)
        else:
            row2.append(i)
    return row1, row2

#calculate inpurity of the hazlenuts dataset
#calculate labels by using uniquerows
#calculate the impurity of each unique value in variety colyms
def giniimpurity(rows):
    count = uniquerows(rows)

    impurity = 0
    for i in count:
        prob = count[i] / float(len(rows))
        impurity = impurity - prob ** 2
    return impurity


#calculate the gain by using giniinpurity
def informationgain(l, r, parent):
    total = (len(l) + len(r))
    x = float(len(l)) / total

    y = float(len(r)) / total
    return parent - x * giniimpurity(l) - y * giniimpurity(r)

#go over the array for find the best split, find the best thresholds
def bestsplit(rows):
    topthreshold = None
    parent = giniimpurity(rows)
    numfeatures = len(rows[0]) - 1
    top_gain = 0

    for column in range(numfeatures):

        row_n = set([row[column] for row in rows])

        for value in row_n:
            threshold = Threshold(column, value)
            row1, row2 = splitrows(rows, threshold)

            if len(row1) == 0 or len(row2) == 0:
                continue

            gain = informationgain(row1, row2, parent)

            if gain >= top_gain:
                top_gain, topthreshold = gain, threshold
    return top_gain, topthreshold

#take note of how many times each unique variety of hazlenut appears
class prediction:
    def __init__(self, rows):
        self.predictions = uniquerows(rows)


class node:
    def __init__(self, threshold, node1, node2):
        self.threshold = threshold
        self.node1 = node1
        self.node2 = node2

#Build a tree based on training data
def buildtree(rows):
    gain, threshold = bestsplit(rows)

    if gain == 0:
        return prediction(rows)

    row1, row2 = splitrows(rows, threshold)

    branch1 = buildtree(row1)
    branch2 = buildtree(row2)

    return node(threshold, branch1, branch2)


def classify(row, node):
    if isinstance(node, prediction):
        return node.predictions

    if node.threshold.compare(row):
        return classify(row, node.node1)

    else:
        return classify(row, node.node2)

#row1 is predicions that were classified correctly
#row2 incorrect
def accuracy(arr, model):
    row1 = []
    row2 = []

    for row in arr:
        if row[-1] in (classify(row, model)):
            row1.append("Classified Correct")
        else:
            row2.append("")

    total = len(row1) + len(row2)
    row1 = len(row1)
    row2 = len(row2)
    return (row1 / total), (row2 / total)

#Run it 10 times and get the average
def test():
    total = []

    for i in range(10):
        test, train = test_train_data("hazlenuts.csv")
        tree = buildtree(train)
        t, f = accuracy(test, tree)
        print("%s: Classified Correct:%.2f" % (str(i + 1), (t * 100)))
        total.append(t)

    print("\nAverage result %.2f percent " % ((sum(total) / len(total) * 100),))


test()
