import threading

class ThreadClass (threading.Thread):
    dictFuncs = dict()
    def __init__(self, threadID, name, owner, func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.dictFuncs[threadID] = (name, owner, func)

    def run(self):
        print('[{0}] starts, id={1}'.format(self.dictFuncs[self.threadID][0],  self.threadID))
        # call thread func 
        self.dictFuncs[self.threadID][2]()

