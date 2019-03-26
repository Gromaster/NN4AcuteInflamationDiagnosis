package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;

public abstract class NeuralLayer {
    protected int numberOfNeuronsInLayer;
    private ArrayList<Neuron> neurons=new ArrayList<>();
    protected IActivationFunction activationFunction;
    protected NeuralLayer previousLayer;
    protected NeuralLayer nextLayer;
    protected ArrayList<Double> input;
    protected ArrayList<Double> output;
    protected int numberOfInputs;




    protected void init(){
        for (int i=0;i<numberOfNeuronsInLayer;i++) {
            try {
                neurons.get(i).setActivationFunction(activationFunction);
                neurons.get(i).init();
            }
            catch(IndexOutOfBoundsException e){
                neurons.add(new Neuron(numberOfInputs,activationFunction));
                neurons.get(i).init();
            }
        }
    }

    void calc(){
        for(int i=0;i<numberOfNeuronsInLayer;i++){
            neurons.get(i).setInput(this.input);
            neurons.get(i).calc();
            try{
                output.set(i,neurons.get(i).getOutput());
            }
            catch (IndexOutOfBoundsException e){
                output.add(neurons.get(i).getOutput());
            }
        }
    }

    NeuralLayer getPreviousLayer() {
        return previousLayer;
    }

    void setPreviousLayer(NeuralLayer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public NeuralLayer getNextLayer() {
        return nextLayer;
    }

    void setNextLayer(NeuralLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public ArrayList<Double> getInput() {
        return input;
    }

    void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    ArrayList<Double> getOutput() {
        return output;
    }

}
