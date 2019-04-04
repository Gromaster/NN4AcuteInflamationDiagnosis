import ActivationFunctions.ActivationFunction;
import ActivationFunctions.IActivationFunction;
import Data.NeuralDataSet;
import NeuralNetworkPackage.NeuralNetwork;


public class Main {


    public static void main(String[] args){
        int numberOfInputs=6;
        int numberOfOutputs=2;
        int numberOfHiddenLayers=2;
        int[] numberOfHiddenNeurons= { 100,100 };





        NeuralDataSet neuralDataSet =new NeuralDataSet(Main.class.getResource("dataset.csv").getPath());





        IActivationFunction[] hiddenAcFnc = {ActivationFunction.SIGMOID} ;
        System.out.println("Creating Neural Network...");
        NeuralNetwork nn = new NeuralNetwork(numberOfInputs, numberOfHiddenLayers,
                numberOfHiddenNeurons,hiddenAcFnc,numberOfOutputs,ActivationFunction.RELU);
        System.out.println("Neural Network created!");

    }
}

