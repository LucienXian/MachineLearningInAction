
# coding: utf-8

# In[1]:


from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


# In[2]:


group, labels = createDataSet()


# In[14]:


group.argsort()


# In[16]:


labels


# In[22]:


array([15,2,3,4]).argsort()


# In[114]:


def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# In[38]:


classify0([0,4],group,labels,3)


# In[46]:


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numOfLines = len(arrayOLines)
    returnMat = zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[:3]
        # classLabelVector.append(int(listFromLine[-1]))
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


# In[80]:


datingdataMat, datingLabels = file2matrix('datingTestSet.txt')
datingdataMat.min(0)


# In[74]:


import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
# s : scalar or array_like, shape (n, ), optional
# c : color, sequence, or sequence of color, optional, default: ‘b’
ax.scatter(datingdataMat[:,0], datingdataMat[:,1],15.0, 10.0*array(datingLabels))
plt.show()


# In[83]:


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # newValue = (oldValue-min)/(max-min)
    normDataSet = (dataSet-tile(minVals, (dataSet.shape[0],1))) / tile(ranges, (dataSet.shape[0], 1))
    return normDataSet, ranges, minVals


# In[85]:


normMat, ranges, minVals = autoNorm(datingdataMat)
normMat


# In[106]:


def datingClassTest():
    hoRation = 0.1
    datingdataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingdataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRation)
    
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], 5)
        print('classifier Result: %d, realAnswer: %d' %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('Error rate: %f' %(errorCount/float(numTestVecs)))


# In[115]:


datingClassTest()


# In[129]:


def classifyPerson():
    resultlist = ['hate', 'a littile like', 'like very much']
    percentTats = float(input('花费在游戏里的时间？'))
    ffMiles = float(input('每年的飞行距离?'))
    iceCreams = float(input('每周的冰淇淋消耗公升数？'))
    datingdataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingdataMat)
    inArr = array([ffMiles, percentTats, iceCreams])
    classfierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('你是否喜欢和该人约会？%s' %resultlist[classfierResult-1])


# In[130]:


classifyPerson()


# In[ ]:


1+1

