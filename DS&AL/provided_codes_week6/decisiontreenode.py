import csv
import math
from voterecord import *

class Node:

    def __init__(self,records):
        self.blnSplit = False
        self.children = {}
        self.records = records
        self.decisionAttribute = -1
        self.stat = {}
        for type in Record.types:
            self.stat[type] = 0
            for record in records:
                if record.party == type:
                    self.stat[type] = self.stat[type] + 1

    def __str__(self):
        if self.blnSplit == True:
            ret = 'Feature '+str(self.decisionAttribute)+ '('+str(self.stat)+')' + '\n'
            for key in self.children.keys():
                ret = ret + '---- key : '+str(key)+ ' ' +str(self.children[key])
        else:
            ret = str(self.stat) + '\n'
        return ret

    def splitNode(self):
        self.blnSplit = True
        gains = self.calculateInformationGainPerFeatures()
        idxMaxGainFeature = -1
        maxGain = -999999999999999999.0
        for itr in range(len(gains)):
            if ?????????? < ??????????:
                ?????????? = ??????????
                idxMaxGainFeature = itr

        for value in Record.values:
            childRecords = []
            for record in self.records:
                if record.feature[idxMaxGainFeature] == value:
                    childRecords.append(record)
            self.children[value] = ??????????(childRecords)
        self.decisionAttribute = idxMaxGainFeature
        return self.children

    def calculateInformationGainPerFeatures(self):
        gains = []
        entropyClass = self.?????
        for itr in range(Record.numValues):
            entropyConditional = self.?????
            gains.append( entropyClass - entropyConditional )
        return gains

    def calculateClassEntropy(self):
        entropy = 0.0
        for type in Record.types:
            cnt = 0.0
            for record in self.records:
                if record.party == type:
                    cnt = cnt + 1.0
            size = float(len(self.records))
            prob = float(????? / ?????)
            entropy = entropy - ????? * math.?????(?????,2)
        return entropy

    def calculateConditionalEntropy(self,idxFeature):
        entropy = 0.0
        for value in Record.values:
            for type in Record.types:
                cntFeature = 0.0
                cntFeatureAndClass = 0.0
                for record in self.records:
                    if record.feature[idxFeature] == value:
                        ????? = ????? + 1
                        if record.party == type:
                            ????? = ????? + 1.0
                size = float(len(self.records))
                probFeature = cntFeature / size + 0.000001
                probFeatureAndClass = cntFeatureAndClass / size + 0.000001
                entropy = entropy + ????? * math.log(?????/?????,2)
        return entropy

if __name__ == "__main__":
    csvfile = open('house-votes-84.csv','rt')
    reader = csv.reader(csvfile,delimiter=',')
    records = []

    for row in reader:
        record = Record(row)
        print (record)
        records.append(record)

    node = Node(records)
    print(node)
    node.splitNode()
    print(node)