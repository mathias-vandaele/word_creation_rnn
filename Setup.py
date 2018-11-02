from UtilsRNN.NeuralNetwork import NeuralNetwork
from Utils.GetDataSet import GetDataSet


if __name__ == '__main__':
    getDataSet = GetDataSet("Data/train.txt")
    features, resultats = getDataSet.getData()
    nn = NeuralNetwork(features, resultats)
    print (nn.loss_calculation())
    for i in range(100):
        nn.backpropagation_throught_time(features[0], resultats[0])
    print (nn.loss_calculation())
