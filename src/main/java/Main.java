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

        ArrayList<Double> neuralInput = new ArrayList<>( Arrays.asList(1.5,0.5) );
        ArrayList<Double> neuralOutput;
        System.out.println("Feeding the values ["+neuralInput.get(0)+" ; "+
                neuralInput.get(1)+"] to the neural network");
        nn.setInput(neuralInput);
        nn.calc();
        neuralOutput= nn.getOutput();
        System.out.println("Output generated"+neuralOutput.toString());

        neuralInput.set(0,1.0);
        neuralInput.set(1,2.1);
        System.out.println("Feeding the values ["+neuralInput.get(0)+" ; "+
                neuralInput.get(1)+"] to the neural network");
        nn.setInput(neuralInput);
        nn.calc();
        neuralOutput=nn.getOutput();
        System.out.println("Output generated"+neuralOutput.toString());
    }
}

