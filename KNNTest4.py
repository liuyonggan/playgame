from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
import numpy as np

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    #打开文件
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    #测试机labels
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%(fileNameStr))
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    error = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult!=classNumber):
            error += 1
    print("总共错了%d个数据\n错误率为%f%%" % (error, error/mTest * 100))



if __name__ == "__main__":
    handwritingClassTest()



