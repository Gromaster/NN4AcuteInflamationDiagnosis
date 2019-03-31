package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;
import java.util.Locale;

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
                    hiddenLayers.add(new HiddenLayer(inputLayer.numberOfNeuronsInLayer, numberOfNeuronsInHiddenLayers[i],
                            (hiddenActivationFunction.length==1) ? hiddenActivationFunction[0] : hiddenActivationFunction[i]));
                    inputLayer.setNextLayer(hiddenLayers.get(i));
                    hiddenLayers.get(i).setPreviousLayer(inputLayer);
                } else {
                    hiddenLayers.add(new HiddenLayer(hiddenLayers.get(i - 1).numberOfNeuronsInLayer,
                            numberOfNeuronsInHiddenLayers[i], hiddenActivationFunction[i]));
                    hiddenLayers.get(i).setPreviousLayer(hiddenLayers.get(i - 1));
                    hiddenLayers.get(i - 1).setNextLayer(hiddenLayers.get(i));
                }
            }

            outputLayer = new OutputLayer(numberOfOutputs, outputActivationFunction,
                    hiddenLayers.get(numberOfHiddenLayers - 1).numberOfNeuronsInLayer);
            outputLayer.setPreviousLayer(hiddenLayers.get(numberOfHiddenLayers - 1));
            hiddenLayers.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);

        } else {
            outputLayer = new OutputLayer(numberOfOutputs, outputActivationFunction, numberOfInputs);

            inputLayer.setNextLayer(outputLayer);

            outputLayer.setPreviousLayer(inputLayer);
        }

        output = new ArrayList<>(numberOfOutputs);
    }

    public void calc() {
        inputLayer.setInputs(input);
        inputLayer.calc();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            HiddenLayer h1 = hiddenLayers.get(i);
            h1.setInputs(h1.getPreviousLayer().getOutputs());
            h1.calc();
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        output = outputLayer.getOutputs();
    }


    public InputLayer getInputLayer() {
        return new InputLayer(inputLayer);
    }

    public void setInputLayer(InputLayer inputLayer) {
        this.inputLayer = new InputLayer(inputLayer);
    }

    public ArrayList<HiddenLayer> getHiddenLayers() {
        ArrayList<HiddenLayer> returnList = new ArrayList<>(hiddenLayers.size());
        for(HiddenLayer h:hiddenLayers)
            returnList.add(new HiddenLayer(h));
        return returnList;
    }

    public void setHiddenLayers(ArrayList<HiddenLayer> hiddenLayers) {
        this.hiddenLayers = new ArrayList<>(hiddenLayers.size());
        for(HiddenLayer h:hiddenLayers)
            this.hiddenLayers.add(new HiddenLayer(h));
    }

    public OutputLayer getOutputLayer() {
        return new OutputLayer(outputLayer);
    }

    public void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = new OutputLayer(outputLayer);
    }

    public int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public void setNumberOfHiddenLayers(int numberOfHiddenLayers) {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public void setNumberOfOutputs(int numberOfOutputs) {
        this.numberOfOutputs = numberOfOutputs;
    }

    public ArrayList<Double> getInput() {
        return new ArrayList<>(input);
    }

    public void setInput(ArrayList<Double> input) {
        this.input = new ArrayList<>(input);
    }

    public ArrayList<Double> getOutput() {
        return new ArrayList<>(output);
    }

    public void setOutput(ArrayList<Double> output) {
        this.output = new ArrayList<>(output);
    }

    public HiddenLayer getHiddenLayers(int layer) {
        return new HiddenLayer(hiddenLayers.get(layer));
    }
}