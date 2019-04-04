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
    private ArrayList<Double> inputs;
    private ArrayList<Double> outputs;

    public NeuralNetwork(int numberOfInputs, int numberOfHiddenLayers, int[] numberOfNeuronsInHiddenLayers,
                         IActivationFunction[] hiddenActivationFunction, int numberOfOutputs,
                         IActivationFunction outputActivationFunction) {

        this.numberOfInputs = numberOfInputs;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.numberOfOutputs = numberOfOutputs;

        this.inputs = new ArrayList<>(numberOfInputs);
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

        outputs = new ArrayList<>(numberOfOutputs);
    }

    public void calc() {
        inputLayer.setInputs(inputs);
        inputLayer.calc();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            HiddenLayer h1 = hiddenLayers.get(i);
            h1.setInputs(h1.getPreviousLayer().getOutputs());
            h1.calc();
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        outputs = outputLayer.getOutputs();
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

    public ArrayList<Double> getInputs() {
        return new ArrayList<>(inputs);
    }

    public Double getInput(int i) {
        return inputs.get(i);
    }

    public void setInputs(ArrayList<Double> inputs) {
        this.inputs = new ArrayList<>(inputs);
    }

    public ArrayList<Double> getOutputs() {
        return new ArrayList<>(outputs);
    }

    public void setOutputs(ArrayList<Double> outputs) {
        this.outputs = new ArrayList<>(outputs);
    }

    public HiddenLayer getHiddenLayer(int i) {
        if(i>=numberOfHiddenLayers) throw new IndexOutOfBoundsException("Illegal argument");
        return new HiddenLayer(hiddenLayers.get(i));
    }

    public void setHiddenLayer(int k, HiddenLayer hiddenLayer) {
        this.hiddenLayers.set(k,new HiddenLayer(hiddenLayer));
    }
}