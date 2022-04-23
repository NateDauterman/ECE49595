
# coding: utf-8

import sys
from numpy import *
from matplotlib import pyplot as plt
import numpy as np
import copy
import csv


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return mat(dataMat)


def loadCenterSet(fileName):      #general function to parse tab -delimited floats
    centerMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        centerMat.append(fltLine)
    return mat(centerMat)


def assignCluster(dataSet, k, centroids):
    '''For each data point, assign it to the closest centroid
    Inputs:
        dataSet: each row represents an observation and
                 each column represents an attribute
        k:  number of clusters
        centroids: initial centroids or centroids of last iteration
    Output:
        clusterAssment: list
            assigned cluster id for each data point
    '''
    clusterAssment = []

    dataSetCopy = copy.deepcopy(dataSet)

    for data in dataSetCopy:
        minDist = -1
        minIndex = -1
        for cenInd, center in enumerate(centroids):
            totalDist = 0
            #print(data, center)
            for dat, cen in zip(np.asarray(data).flatten(), np.asarray(center).flatten()):
                #print(dat, cen)
                totalDist += (dat - cen) ** 2
            totalDist = sqrt(totalDist)
            if minDist == -1 or totalDist < minDist:
                minDist = totalDist
                minIndex = cenInd
        clusterAssment.append(minIndex)

    return clusterAssment


def getCentroid(dataSet, k, clusterAssment):
    '''recalculate centroids
    Input:
        dataSet: each row represents an observation and
            each column represents an attribute
        k:  number of clusters
        clusterAssment: list
            assigned cluster id for each data point
    Output:
        centroids: cluster centroids
    '''

    dataSetCopy = copy.deepcopy(dataSet)

    centroids = []
    #print(centroids)

    lastIndex = max(clusterAssment)
    for cenInd in range(lastIndex + 1):
        #print(cenInd)
        indexes = np.where(np.array(clusterAssment) == cenInd)[0]
        #print(indexes)
        totalPoints = 0
        for ind in indexes:
            if isinstance(totalPoints, int):
                totalPoints = copy.deepcopy(dataSet[ind])
            else:
                totalPoints += dataSet[ind]
        #print(totalPoints)
        totalPoints /= len(indexes)
        #print(totalPoints.tolist()[0])
        centroids.append(totalPoints.tolist()[0])
    centroids = np.matrix(centroids)

    return centroids


def kMeans(dataSet, T, k, centroids):
    '''
    Input:
        dataSet: each row represents an observation and
                each column represents an attribute
        T:  number of iterations
        k:  number of clusters
        centroids: initial centroids
    Output:
        centroids: final cluster centroids
        clusterAssment: list
            assigned cluster id for each data point
    '''
    clusterAssment = [0] * len(dataSet)
    pre_clusters  = [1] * len(dataSet)

    i=1
    while i < T and list(pre_clusters) != list(clusterAssment):
        pre_clusters = copy.deepcopy(clusterAssment)
        clusterAssment = assignCluster(dataSet, k, centroids )
        centroids      = getCentroid(dataSet, k, clusterAssment)
        i=i+1

    return centroids, clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        data_filename = sys.argv[1]
        centroid_filename = sys.argv[2]
        k = int(sys.argv[3])
    else:
        data_filename = 'Iris.csv'
        centroid_filename = 'Iris_Initial_Centroids.csv'
        k = 3

    save_filename = data_filename.replace('.csv', '_kmeans_cluster.csv')

    data = loadDataSet(data_filename)
    centroids = loadCenterSet(centroid_filename)
    centroids, clusterAssment = kMeans(data, 7, k, centroids )
    print(centroids)
    saveData(save_filename, data, clusterAssment)


    ### Example: python kmeans_template.py Iris.csv Iris_Initial_Centroids.csv
