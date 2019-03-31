package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

 class OutputLayer extends HiddenLayer{
     OutputLayer(int numberOfNeurons, IActivationFunction activationFunction,int numberOfInputs){
        super(numberOfInputs,numberOfNeurons,activationFunction);
        setNextLayer(null);
        init();
    }

     public OutputLayer(OutputLayer original) {
         super(original);
     }

     public Neuron getNeuron(int neuron){
         if(neuron>=neurons.size())throw new ArrayIndexOutOfBoundsException();
         else return neurons.get(neuron);
    }

}
