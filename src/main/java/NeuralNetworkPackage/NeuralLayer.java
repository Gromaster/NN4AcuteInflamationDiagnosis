package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;

public abstract class NeuralLayer {
     int numberOfNeuronsInLayer;
     ArrayList<Neuron> neurons;
     IActivationFunction activationFunction;
     private NeuralLayer previousLayer;
     private NeuralLayer nextLayer;
     private ArrayList<Double> inputs;
     private ArrayList<Double> outputs;
     int numberOfInputs;

    public NeuralLayer(NeuralLayer origin) {
        setNumberOfNeuronsInLayer(origin.getNumberOfNeuronsInLayer());
        setNeurons(origin.getNeurons());
        setActivationFunction(origin.getActivationFunction());
        setPreviousLayer(origin.getPreviousLayer());
        setNextLayer(origin.getNextLayer());
        setInputs(origin.getInputs());
        setOutputs(origin.getOutputs());
        setNumberOfInputs(origin.getNumberOfInputs());
    }

    NeuralLayer(int numberOfInputs, int numberOfNeuronsInLayer, IActivationFunction activationFunction){
        this.setNumberOfNeuronsInLayer(numberOfNeuronsInLayer);
        neurons=new ArrayList<>(numberOfNeuronsInLayer);
        this.activationFunction=activationFunction;
        this.numberOfInputs=numberOfInputs;
    }

    protected void init() {
        for (int i=0;i<numberOfNeuronsInLayer;i++) {
            try {
                neurons.get(i).setActivationFunction(activationFunction);
                neurons.get(i).init();
            }
            catch(IndexOutOfBoundsException | NullPointerException e) {
                neurons.add(new Neuron(numberOfInputs,activationFunction));
                neurons.get(i).init();
            }
        }
    }

    void calc(){
        for(int i=0;i<numberOfNeuronsInLayer;i++){
            neurons.get(i).setInputs(inputs);
            neurons.get(i).calc();
            try{
                outputs.set(i,neurons.get(i).getOutput());
            }
            catch (IndexOutOfBoundsException | NullPointerException e){
                outputs.add(neurons.get(i).getOutput());
            }
        }
    }

    public int getNumberOfNeuronsInLayer() {
        return numberOfNeuronsInLayer;
    }

    public void setNumberOfNeuronsInLayer(int numberOfNeuronsInLayer) {
        this.numberOfNeuronsInLayer = numberOfNeuronsInLayer;
    }

    public ArrayList<Neuron> getNeurons() {
        ArrayList<Neuron> returnList=new ArrayList<>(neurons.size());
        for(Neuron n:neurons)
            returnList.add(new Neuron(n));
        return returnList;
    }

    public void setNeurons(ArrayList<Neuron> neurons) {
        for(Neuron n:neurons)
            this.neurons.add(new Neuron(n));
        this.neurons = neurons;
    }

    public IActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public NeuralLayer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(NeuralLayer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public NeuralLayer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(NeuralLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public ArrayList<Double> getInputs() {
        return new ArrayList<>(inputs);
    }

    public void setInputs(ArrayList<Double> input) {
        this.inputs = new ArrayList<>(input);
    }

    public ArrayList<Double> getOutputs() {
        return new ArrayList<>(outputs);
    }

    public void setOutputs(ArrayList<Double> output) {
        this.outputs = new ArrayList<>(output);
    }

    public double getWeight(int i, int j) {
        return neurons.get(i).getWeights().get(j);
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }


    public Neuron getNeuron(int i) {
        return new Neuron(neurons.get(i));
    }
}
