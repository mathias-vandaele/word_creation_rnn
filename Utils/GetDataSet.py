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
        nextLetter.append(32)

        return np.reshape(letter[:1000], (100,10)), np.reshape(nextLetter[:1000], (100,10))

    def get_dictonnary(self):
        features = []
        targets = []
        for line in self.fileObj:
            letters = []
            next_letters = []
            word = []

            for ch in line:
                word.append(ord(ch))

            next_letters = copy.deepcopy(word)
            letters = copy.deepcopy(word)
            del letters[-1]
            del next_letters[0]
            features.append(letters)
            targets.append(next_letters)

        return features, targets
