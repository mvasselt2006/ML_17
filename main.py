#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("mfeat-pix.txt").reshape(2000 ,240)

colors = [ "\033[38;2;97;134;118m","\033[38;2;107;144;128m", "\033[38;2;164;195;178m","\033[38;2;204;227;222m","\033[38;2;234;244;244m","\033[38;2;244;255;248m","\033[38;2;255;255;255m" ]

#Split data
trainPatterns = np.zeros((1000,240))
testPatterns = np.zeros((1000,240))
for i in range(10):
    trainPatterns[i*100:(i+1)*100,:]=data[i*200:(i*200+100),:]
    testPatterns[i*100:(i+1)*100,:]=data[(i*200+100):(i*200+200),:]

#generate the labels
#create indicator matrices size 10 x 1000 with the class labels coded by 
#binary indicator vectors
b = np.ones((1,100))
trainLabels = np.zeros((10,1000))
for i in range(10):
    trainLabels[i,(i*100):((i+1)*100)]=1

testLabels = trainLabels

correctLabels = np.concatenate((b, 2*b, 3*b, 4*b, 5*b, 6*b, 7*b, 8*b, 9*b, 10*b))

def show_datapoint(datapoint):
    for i in range(16):
        row = ""
        for j in range(15):
            row = row + colors[int(datapoint[i*15+j])] + "▓"
        row = row + "\033[0m"
        print(row)

# 15 wide, 16 high

def normdot(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def gen_mean_prototypes(data, labels):
    labels = labels.flatten()
    proto = []
    for i in range(10):
        indices = np.where(labels == (i+1))[0]
        proto.append(np.mean(data[indices,:], axis=0))
    return proto

#Prototype matching
def match_prototypes(data, labels, prototypes):
    """function which returns a matrix where each column shows the similarity with each prototype.
    The prototype taken is the mean of each class"""
    labels=labels.flatten()
    match =  np.zeros((1000,10))
    for i in range(10):
        proto = prototypes[i]
        for k in range(1000):
            match[k, i] = normdot(proto, data[k])
    
    return match


def display_protoypes(prototype):
    for i in range(10):
        show_datapoint(prototype[i])

def scatter_prototype(prototype, x, y):
    scatter = np.zeros((10, 100, 2))
    for j in range(1000):
        match_x = normdot(trainPatterns[j], prototype[x])
        match_y = normdot(trainPatterns[j], prototype[y])
        scatter[j // 100, j % 100, 0] = match_x
        scatter[j // 100, j % 100, 1] = match_y

    for digit in range(10):
        plt.scatter(scatter[digit, :, 0], scatter[digit, :, 1], label=str(digit))
    plt.legend()
    plt.show()


def eval_prototype(prototype, name, display_wrongs=False):
    # todo move to parameter: correctLabels, trainPatterns
    #Check output
    match = match_prototypes(testPatterns, correctLabels, prototype)
    inx = np.argmax(match, axis=1)
    #Check how many digits get a different best matched prototype than their label
    diff_proto= (correctLabels.flatten()-(inx+1))
    numb_diff_proto= np.size((np.where(diff_proto!=0)[0]))
    for j in range(1000):
        if diff_proto[j] != 0:
            show_datapoint(testPatterns[j])
            print("Incorrect classification: ", inx[j])
    print("Amount of digits which don't have the closest match with the correct prototype for the prototype " + name,numb_diff_proto)

mean_prototypes = gen_mean_prototypes(trainPatterns, correctLabels)
eval_prototype(mean_prototypes, "mean", True)
#display_protoypes(mean_prototypes)
scatter_prototype(mean_prototypes, 0, 8)

# find better prototypes?

def evaluate(pattern, selection):
    match = 0
    for j in range(len(selection)):
        match += normdot(pattern, selection[j])
    return match / len(selection)

def gen_chosen_prototypes(data, labels):
    labels = labels.flatten()
    proto = []
    for i in range(10):
        indices = np.where(labels == (i+1))[0]
        selection = data[indices,:]
        max_reached = 0
        max_index = 0
        for j in range(len(selection)):
            c = evaluate(selection[j], selection)
            if c > max_reached:
                max_reached = c
                max_index = j
        proto.append(selection[max_index])
    return proto

#chosen_prototypes = gen_chosen_prototypes(trainPatterns, correctLabels)
#eval_prototype(chosen_prototypes, "chosen")
#scatter_prototype(chosen_prototypes, 2, 5)


def scatter_slope_curvature(data, labels, left=True, top=True):
    # dims: index, (x,y = slope,curvature)
    scatter = np.zeros((10, 100, 2))

    for j in range(1000):
        # 16 rows
        avgs = np.zeros(8)
        for k in range(8):
            row = k if top else k + 8
            avg = 0.0
            valsum = 0.0
            for l in range(7):
                column = l if left else l + 8
                valsum += data[j, row * 15 + column] / 6
                avg += column * data[j, row * 15 + column] / 6
            avg = avg / valsum
            avgs[k] = avg
        slopes = np.gradient(avgs)
        curves = np.gradient(slopes)
        slope = np.mean(slopes)
        curve = np.mean(curves)

        if j == 0:
            show_datapoint(data[j])
            print(avgs)
            print(slopes)
            print(curves)

        scatter[j // 100, j % 100, 0] = slope
        scatter[j // 100, j % 100, 1] = curve

    for digit in range(10):
        plt.scatter(scatter[digit, :, 0], scatter[digit, :, 1], label=str(digit))

    left_text = "left" if left else "right"
    top_text = "top" if top else "bottom"
    plt.title("Quadrant " + top_text + " " + left_text)
    plt.xlabel("Slope")
    plt.ylabel("Curvature")
    plt.legend()
    plt.figure()


#scatter_slope_curvature(trainPatterns, correctLabels, True, True)
#scatter_slope_curvature(trainPatterns, correctLabels, True, False)
#scatter_slope_curvature(trainPatterns, correctLabels, False, True)
#scatter_slope_curvature(trainPatterns, correctLabels, False, False)


#plt.show()

