#--coding:UTF-8--
import operator

import numpy as np
def classify0(inX,dataSet,labels,k):
    dataSetsize = dataSet.shape[0]
    diffmat= np.tile(inX,(dataSetsize,1))-dataSet
    sqDiffMat = diffmat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    #打开文档
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    #返回的分类标签向量
    classLableVestor = []
    index = 0
    for line in arrayOfLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLableVestor.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLableVestor.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLableVestor.append(3)
        index += 1

    return returnMat, classLableVestor

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals - minVals
    normDataSet = np.zeros((np.shape(dataSet)))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(range,(m,1))
    return normDataSet,range,minVals


# def datingClasstest():
#     filename = 'datingTestSet.txt'
#     datingDataMat,datingLables = file2matrix(filename)
#     hoRatio = 0.1
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     m = normMat.shape[0]
#     numTestVecs = int(m*hoRatio)
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLables[numTestVecs:m], 4)
#         print("分类结果：%d\t真实分类：%d"%(classifierResult,datingLables[i]))
#         if classifierResult != datingLables[i]:
#             errorCount += 1.0
#     print("错误率：%f%%"%(errorCount/float(numTestVecs)*100))
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))


if __name__ =="__main__":
    classifyPerson()

