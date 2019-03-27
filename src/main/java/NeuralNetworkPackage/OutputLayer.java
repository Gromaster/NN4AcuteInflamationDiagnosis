package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

 class OutputLayer extends HiddenLayer{
     OutputLayer(int numberOfNeurons, IActivationFunction activationFunction,int numberOfInputs){
        super(numberOfInputs,numberOfNeurons,activationFunction);
        setNextLayer(null);
        init();
    }
}
