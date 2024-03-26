class AssempblyLines:
    timeStation = [[7,9,3,4,8,4],[8,5,6,4,5,7]]
    timeBelt = [[2,2,3,1,3,4,3],[4,2,1,2,2,1,2]]

    timeScheduling =[list(range(6)), list(range(6))]
    stationTracing = [list(range(6)), list(range(6))]


    intCont = 0
    def SchedulingRC(self, idxLine, idxStation):
        self.intCont += 1
        if idxStation == 0:
            if idxLine == 1:
                return self.timeStation[0][0] + self.timeBelt[0][0]
            elif idxLine == 2:
                return self.timeStation[1][0] + self.timeBelt[1][0]
        
        if idxLine == 1:
            costLine1 = self.SchedulingRC(1 , idxStation - 1) + self.timeStation[0][idxStation]
            costLine2 = self.SchedulingRC(2, idxStation -1 ) + self.timeBelt[1][idxStation] + self.timeStation[0][idxStation]
        elif idxLine ==2:
            costLine1 = self.SchedulingRC(1, idxStation -1) + self.timeStation[1][idxStation] + self.timeBelt[0][idxStation]
            costLine2 = self.SchedulingRC(2, idxStation - 1) + self.timeStation[1][idxStation]
        
        if costLine1 > costLine2:
            return costLine2
        else:
            return costLine1


    def startSchedulingRC(self):
        numStation = len(self.timeStation[0])
        costLine1 = self.SchedulingRC(1, numStation - 1) + self.timeBelt[0][numStation]
        costLine2 = self.SchedulingRC(2, numStation - 1) + self.timeBelt[1][numStation]
        if costLine1 < costLine2:
            return costLine1
        else:
            return costLine2


    def startSchedulingDP(self):
        numStation = len(self.timeStation[0])
        self.timeScheduling[0][0] = self.timeStation[0][0] + self.timeBelt[0][0]
        self.timeScheduling[1][0] = self.timeStation[1][0] + self.timeBelt[1][0]

        for i in range(1,numStation):

            if self.timeScheduling[0][i -1] < self.timeScheduling[1][i-1] + self.timeBelt[1][i]:
                #그냥이 빠를때
                self.timeScheduling[0][i] = self.timeScheduling[0][i-1] + self.timeStation[0][i]
                self.stationTracing[0][i] = 0
            else:
                #아래에서 오는게 빠를때
                self.timeScheduling[0][i] = self.timeScheduling[1][i-1] + self.timeBelt[1][i] + self.timeStation[0][i]
                self.stationTracing[0][i] = 1

            if self.timeScheduling[1][i-1] < self.timeScheduling[0][i-1] + self.timeBelt[0][i]:
                self.timeScheduling[1][i] = self.timeScheduling[1][i-1] + self.timeStation[1][i]
                self.stationTracing[1][i] = 1
            else:
                #위에서 오는게 빠를때
                self.timeScheduling[1][i] = self.timeScheduling[0][i-1] + self.timeBelt[0][i] + self.timeStation[1][i]
                self.timeStation[1][i] = 0
            
        costLine1= self.timeScheduling[0][numStation-1] + self.timeBelt[0][numStation]
        costLine2 = self.timeScheduling[1][numStation-1] + self.timeBelt[1][numStation]
        if costLine1 > costLine2:
            return costLine2, 1
        else:
            return costLine1, 0
    def printTracing(self, lineTracing):
        numStation = len(self.timeStation[0])
        print("Line : ",lineTracing)
        for itr in range(numStation-1, 0, -1):
            lineS = self.stationTracing[lineTracing][itr]
            print(lineS)

lines = AssempblyLines()
#time, lineTracing = lines.startSchedulingDP()
#print(time)
#lines.printTracing(lineTracing)

times = lines.startSchedulingRC()
print(times)