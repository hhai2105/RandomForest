from utils import *
from decisiontree import *
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numba

numba.jit

def wine(ntree, numFolds, isGini, minInfoGainMaxGiniIndex):
    numFolds = 10
    dataset = pd.read_csv('./datasets/hw3_wine.csv', sep="\t",header=0)

    #moving class to the end
    classColumn = dataset.pop("# class")
    dataset.insert(len(dataset.columns),"# class", classColumn)
    dataset = dataset.to_records(index=False)

    #create datatype array
    typeArray = []
    for i in range(len(dataset[0]) - 1):
        typeArray.append("numerical")

    #stratifiedkfolds
    folds = stratifiedKFold(dataset, numFolds)

    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    for i in range(numFolds):
        forest = []
        testingSet = folds[i]
        trainingSet = []
        for j in range(0, i):
            trainingSet += folds[j]
        for j in range(i + 1, numFolds):
            trainingSet += folds[j]
        forest = createForest( ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex)
        accuracy, precision, recall, f1 = testing(forest, testingSet)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1 
    averageAccuracy =  totalAccuracy / numFolds
    averagePrecision = totalPrecision / numFolds
    averageRecall = totalRecall / numFolds
    averageF1 = totalF1 / numFolds
    print("accuracy is:" + str(averageAccuracy))
    print("precision is:" + str(averagePrecision))
    print("recall is:" + str(averageRecall))
    print("F1 is:" + str(averageF1))
    print("----------------------------------------")
    return averageAccuracy, averagePrecision, averageRecall, averageF1
    
def houseVote(ntree, numFolds, isGini, minInfoGainMaxGiniIndex):
    dataset = pd.read_csv('./datasets/hw3_house_votes_84.csv', sep=",",header=0)

    #moving class to the end
    dataset = dataset.to_records(index=False)

    skf = StratifiedKFold(n_splits=10)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

    #create datatype array
    typeArray = []
    for i in range(len(dataset[0]) - 1):
        typeArray.append("categorical")

    #stratifiedkfolds
    folds = stratifiedKFold(dataset, numFolds)

    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    #for i in range(10):
    for i in range(numFolds):
        forest = []
        testingSet = folds[i]
        trainingSet = []
        for j in range(0, i):
            trainingSet += folds[j]
        for j in range(i + 1, 10):
            trainingSet += folds[j]
        forest = createForest( ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex)
        testing(forest, testingSet)
        accuracy, precision, recall, f1 = testing(forest, testingSet)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1 
    averageAccuracy =  totalAccuracy / numFolds
    averagePrecision = totalPrecision / numFolds
    averageRecall = totalRecall / numFolds
    averageF1 = totalF1 / numFolds
    print("accuracy is:" + str(averageAccuracy))
    print("precision is:" + str(averagePrecision))
    print("recall is:" + str(averageRecall))
    print("F1 is:" + str(averageF1))
    print("----------------------------------------")
    return averageAccuracy, averagePrecision, averageRecall, averageF1

def runWine(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    ntreeValues = [1, 2, 10, 20, 30, 40, 50]
    for i in range(len(ntreeValues)):
        newAccuracy, newPrecision, newRecall, newF1 = wine(ntreeValues[i], 10, isGini, minInfoGainMaxGiniIndex)
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    plt.plot(ntreeValues, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of trees')
    plt.title('correlation between accuracy and number of trees (wine)')
    plt.figure()
    plt.plot(ntreeValues, precision)
    plt.ylabel('precision')
    plt.xlabel('number of trees')
    plt.title('correlation between precision and number of trees (wine)')
    plt.figure()
    plt.plot(ntreeValues, recall)
    plt.ylabel('recall')
    plt.xlabel('number of trees')
    plt.title('correlation between recall and number of trees (wine)')
    plt.figure()
    plt.plot(ntreeValues, f1)
    plt.ylabel('f1')
    plt.xlabel('number of trees')
    plt.title('correlation between f1 score and number of trees (wine)')
    plt.show()

def runHouseVote(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    ntreeValues = [1, 2, 10, 20, 30, 40, 50]
    for i in range(len(ntreeValues)):
        newAccuracy, newPrecision, newRecall, newF1 = houseVote(ntreeValues[i], 10, isGini, minInfoGainMaxGiniIndex)
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    plt.plot(ntreeValues, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of trees')
    plt.title('correlation between accuracy and number of trees (houseVote)')
    plt.figure()
    plt.plot(ntreeValues, precision)
    plt.ylabel('precision')
    plt.xlabel('number of trees')
    plt.title('correlation between precision and number of trees (houseVote)')
    plt.figure()
    plt.plot(ntreeValues, recall)
    plt.ylabel('recall')
    plt.xlabel('number of trees')
    plt.title('correlation between recall and number of trees (houseVote)')
    plt.figure()
    plt.plot(ntreeValues, f1)
    plt.ylabel('f1')
    plt.xlabel('number of trees')
    plt.title('correlation between f1 score and number of trees (houseVote)')

def cancer(ntree, numFolds, isGini, minInfoGainMaxGiniIndex):
    dataset = pd.read_csv('./datasets/hw3_cancer.csv', sep="\t",header=0)

    #moving class to the end
    dataset = dataset.to_records(index=False)

    skf = StratifiedKFold(n_splits=10)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

    #create datatype array
    typeArray = []
    for i in range(len(dataset[0]) - 1):
        typeArray.append("numerical")

    #stratifiedkfolds
    folds = stratifiedKFold(dataset, numFolds)

    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    #for i in range(10):
    for i in range(numFolds):
        forest = []
        testingSet = folds[i]
        trainingSet = []
        for j in range(0, i):
            trainingSet += folds[j]
        for j in range(i + 1, 10):
            trainingSet += folds[j]
        forest = createForest( ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex)
        testing(forest, testingSet)
        accuracy, precision, recall, f1 = testing(forest, testingSet)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1 
    averageAccuracy =  totalAccuracy / numFolds
    averagePrecision = totalPrecision / numFolds
    averageRecall = totalRecall / numFolds
    averageF1 = totalF1 / numFolds
    print("accuracy is:" + str(averageAccuracy))
    print("precision is:" + str(averagePrecision))
    print("recall is:" + str(averageRecall))
    print("F1 is:" + str(averageF1))
    print("----------------------------------------")
    return averageAccuracy, averagePrecision, averageRecall, averageF1

def runCancer(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    ntreeValues = [1, 2, 10, 20, 30, 40, 50]
    for i in range(len(ntreeValues)):
        newAccuracy, newPrecision, newRecall, newF1 = cancer(ntreeValues[i], 10, isGini, 1)
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    plt.plot(ntreeValues, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of trees')
    plt.title('correlation between accuracy and number of trees (cancer)')
    plt.figure()
    plt.plot(ntreeValues, precision)
    plt.ylabel('precision')
    plt.xlabel('number of trees')
    plt.title('correlation between precision and number of trees (cancer)')
    plt.figure()
    plt.plot(ntreeValues, recall)
    plt.ylabel('recall')
    plt.xlabel('number of trees')
    plt.title('correlation between recall and number of trees (cancer)')
    plt.figure()
    plt.plot(ntreeValues, f1)
    plt.ylabel('f1')
    plt.xlabel('number of trees')
    plt.title('correlation between f1 score and number of trees (cancer)')
    plt.show()

def contraceptive(ntree, numFolds, isGini):
    dataset = pd.read_csv('./datasets/cmc.data', sep=",",header=None)

    #moving class to the end
    dataset = dataset.to_records(index=False)

    #create datatype array
    typeArray = []
    typeArray.append("numerical")
    typeArray.append("categorical")
    typeArray.append("categorical")
    typeArray.append("numerical")
    typeArray.append("categorical")
    typeArray.append("categorical")
    typeArray.append("categorical")
    typeArray.append("categorical")
    typeArray.append("categorical")

    #stratifiedkfolds
    folds = stratifiedKFold(dataset, numFolds)

    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    #for i in range(10):
    for i in range(numFolds):
        forest = []
        testingSet = folds[i]
        trainingSet = []
        for j in range(0, i):
            trainingSet += folds[j]
        for j in range(i + 1, 10):
            trainingSet += folds[j]
        forest = createForest( ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex)
        testing(forest, testingSet)
        accuracy, precision, recall, f1 = testing(forest, testingSet)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1 
    averageAccuracy =  totalAccuracy / numFolds
    averagePrecision = totalPrecision / numFolds
    averageRecall = totalRecall / numFolds
    averageF1 = totalF1 / numFolds
    print("accuracy is:" + str(averageAccuracy))
    print("precision is:" + str(averagePrecision))
    print("recall is:" + str(averageRecall))
    print("F1 is:" + str(averageF1))
    print("----------------------------------------")
    return averageAccuracy, averagePrecision, averageRecall, averageF1
    
def runContraceptive(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    ntreeValues = [1, 2, 10, 20, 30, 40, 50]
    for i in range(len(ntreeValues)):
        newAccuracy, newPrecision, newRecall, newF1 = contraceptive(ntreeValues[i], 10, isGini, minInfoGainMaxGiniIndex)
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    plt.plot(ntreeValues, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of trees')
    plt.title('correlation between accuracy and number of trees (contraceptive)')
    plt.figure()
    plt.plot(ntreeValues, precision)
    plt.ylabel('precision')
    plt.xlabel('number of trees')
    plt.title('correlation between precision and number of trees (contraceptive)')
    plt.figure()
    plt.plot(ntreeValues, recall)
    plt.ylabel('recall')
    plt.xlabel('number of trees')
    plt.title('correlation between recall and number of trees (contraceptive)')
    plt.figure()
    plt.plot(ntreeValues, f1)
    plt.ylabel('f1')
    plt.xlabel('number of trees')
    plt.title('correlation between f1 score and number of trees (contraceptive)')
    plt.show()


runWine(1)
#runCancer(0)
#runHouseVote(1)
#runContraceptive(0)
