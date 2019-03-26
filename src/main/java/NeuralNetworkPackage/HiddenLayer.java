package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

public class HiddenLayer extends NeuralLayer{
    public HiddenLayer(int numberOfNeurons, IActivationFunction activationFunction,int numberOfInputs){
        this.numberOfNeuronsInLayer=numberOfNeurons;
        this.activationFunction=activationFunction;
        this.numberOfInputs=numberOfInputs;
        init();
    }
}
