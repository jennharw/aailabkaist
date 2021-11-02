class PriorityNode:
    value = ""
    priority = -1
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

    def getValue(self):
        return self.value
    def getPriority(self):
        return self.priority
