package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

 public class HiddenLayer extends NeuralLayer {
     HiddenLayer(int numberOfInputs, int numberOfNeurons, IActivationFunction activationFunction) {
         super(numberOfInputs, numberOfNeurons, activationFunction);
         init();
     }

     public HiddenLayer(NeuralLayer original) {
         super(original);
     }



 }
