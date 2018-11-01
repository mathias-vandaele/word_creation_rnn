import copy
import numpy as np

class GetDataSet:
    """docstring for GetDataSet."""

    def __init__(self, fileName):
        super(GetDataSet, self).__init__()
        self.fileObj = open(fileName, "r")

    def readFile(self):
        """
            Permits to see the text we are going to work on
        """
        letter = []
        for line in self.fileObj:
            for ch in line:
                letter.append(ch)

        nextLetter = copy.deepcopy(letter)
        nextLetter.pop(0)
        nextLetter.append("\n")

        print (len(nextLetter))
        print (len(letter))
        print (letter[:10])
        print (nextLetter[:10])

    def getData(self):
        letter = []
        for line in self.fileObj:
            for ch in line:
                letter.append(ord(ch))

        nextLetter = copy.deepcopy(letter)
        nextLetter.pop(0)
        nextLetter.append("\n")
        return np.reshape(letter[:10], (1,-1)), np.reshape(nextLetter[:10], (1,-1))
