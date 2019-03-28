import ActivationFunctions.ActivationFunction;
import ActivationFunctions.IActivationFunction;
import NeuralNetworkPackage.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {


    public static void main(String[] args){
        int numberOfInputs=2;
        int numberOfOutputs=1;
        int numberOfHiddenLayers=1;
        int[] numberOfHiddenNeurons= { 10 };
        IActivationFunction[] hiddenAcFnc = { ActivationFunction.SIGMOID } ;
        System.out.println("Creating Neural Network...");
        NeuralNetwork nn = new NeuralNetwork(numberOfInputs, numberOfHiddenLayers,
                numberOfHiddenNeurons,hiddenAcFnc,numberOfOutputs,ActivationFunction.RELU);
        System.out.println("Neural Network created!");


    }
}

