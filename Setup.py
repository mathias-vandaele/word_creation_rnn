from UtilsRNN.NeuralNetwork import NeuralNetwork
from Utils.GetDataSet import GetDataSet


if __name__ == '__main__':
    getDataSet = GetDataSet("Data/train.txt")
    features, resultats = getDataSet.getData()
    #print (features, resultats)
    nn = NeuralNetwork(features[:10], resultats[:10])
    pred = nn.backpropagation_throught_time()
