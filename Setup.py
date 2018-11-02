from UtilsRNN.NeuralNetwork import NeuralNetwork
from Utils.GetDataSet import GetDataSet


if __name__ == '__main__':
    getDataSet = GetDataSet("Data/liste_francais.txt")
    features, resultats = getDataSet.get_dictonnary()
    nn = NeuralNetwork(features, resultats)

    print (nn.loss_calculation())
    for i in range(10):
        if i % 2 == 0:
            print (nn.loss_calculation())
        nn.bptt_througt_all_dataset()
    print (nn.loss_calculation())

    print (nn.generate_word(98, 5))
