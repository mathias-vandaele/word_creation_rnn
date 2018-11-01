from UtilsRNN.NeuralNetwork import NeuralNetwork
from Utils.GetDataSet import GetDataSet


if __name__ == '__main__':
    getDataSet = GetDataSet("Data/train.txt")
    features, resultats = getDataSet.getData()
    nn = NeuralNetwork(features[0], resultats[0])
    pred = nn.backpropagation_throught_time()
