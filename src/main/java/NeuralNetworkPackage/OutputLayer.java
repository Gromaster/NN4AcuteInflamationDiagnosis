package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

public class OutputLayer extends HiddenLayer{
    public OutputLayer(int numberOfNeurons, IActivationFunction activationFunction,int numberOfInputs){
        super(numberOfNeurons,activationFunction,numberOfInputs);
        setNextLayer(null);
        init();
    }
}
