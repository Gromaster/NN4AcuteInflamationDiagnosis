package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;

public class NeuralNetwork {
    private InputLayer inputLayer;
    private ArrayList<HiddenLayer> hiddenLayers;
    private OutputLayer outputLayer;
    private int numberOfHiddenLayers;
    private int numberOfInputs;
    private int numberOfOutputs;
    private ArrayList<Double> input;
    private ArrayList<Double> output;

    public NeuralNetwork(int numberOfInputs, int numberOfHiddenLayers, int[] numberOfNeuronsInHiddenLayers,
                         IActivationFunction[] hiddenActivationFunction, int numberOfOutputs,
                         IActivationFunction outputActivationFunction) {

        this.numberOfInputs = numberOfInputs;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.numberOfOutputs = numberOfOutputs;

        this.input = new ArrayList<>(numberOfInputs);
        this.inputLayer = new InputLayer(numberOfInputs);

        if (numberOfHiddenLayers > 0) {
            this.hiddenLayers = new ArrayList<>();
            for (int i = 0; i < numberOfHiddenLayers; i++) {
                if (i == 0) {
                    hiddenLayers.add(new HiddenLayer(numberOfNeuronsInHiddenLayers[i], hiddenActivationFunction[i],
                            numberOfInputs));
                    inputLayer.setNextLayer(hiddenLayers.get(i));
                    hiddenLayers.get(i).setPreviousLayer(inputLayer);
                } else {
                    hiddenLayers.add(new HiddenLayer(numberOfNeuronsInHiddenLayers[i], hiddenActivationFunction[i],
                            hiddenLayers.get(i - 1).numberOfNeuronsInLayer));
                    hiddenLayers.get(i).setPreviousLayer(hiddenLayers.get(i - 1));
                    hiddenLayers.get(i - 1).setNextLayer(hiddenLayers.get(i));
                }
            }
            outputLayer = new OutputLayer(numberOfOutputs, outputActivationFunction,
                    hiddenLayers.get(numberOfHiddenLayers - 1).numberOfNeuronsInLayer);
            outputLayer.setPreviousLayer(hiddenLayers.get(numberOfHiddenLayers - 1));
        } else {
            outputLayer = new OutputLayer(numberOfOutputs, outputActivationFunction, numberOfInputs);
            inputLayer.setNextLayer(outputLayer);
            outputLayer.setPreviousLayer(inputLayer);
        }

        output = new ArrayList<>(numberOfOutputs);
    }

    public void calc() {
        inputLayer.setInput(input);
        inputLayer.calc();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            HiddenLayer h1 = hiddenLayers.get(i);
            h1.setInput(h1.getPreviousLayer().getOutput());
            h1.calc();
        }
        outputLayer.setInput(outputLayer.getPreviousLayer().getOutput());
        outputLayer.calc();
        this.output = outputLayer.getOutput();
    }

    public ArrayList<Double> getOutput() {
        return output;
    }

    public void setInput(ArrayList<Double> input) {
        this.input = input;
    }
}