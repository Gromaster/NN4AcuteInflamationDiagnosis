package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;
import java.util.Collections;

public abstract class NeuralLayer {
     int numberOfNeuronsInLayer;
     ArrayList<Neuron> neurons;
     IActivationFunction activationFunction;
     private NeuralLayer previousLayer;
     private NeuralLayer nextLayer;
     private ArrayList<Double> input;
     private ArrayList<Double> output;
     int numberOfInputs;


     NeuralLayer(int numberOfInputs,int numberOfNeuronsInLayer,IActivationFunction activationFunction){
        this.numberOfNeuronsInLayer=numberOfNeuronsInLayer;
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
            neurons.get(i).setInput(this.input);
            neurons.get(i).calc();
            System.out.println("Neuron nr "+i+" output: "+neurons.get(i).getOutput());
            try{
                System.out.println(neurons.toString());

                output.set(i,neurons.get(i).getOutput());
            }
            catch (IndexOutOfBoundsException | NullPointerException e){
                output.add(neurons.get(i).getOutput());
            }
        }
    }

    NeuralLayer getPreviousLayer() {
        return previousLayer;
    }

    void setPreviousLayer(NeuralLayer previousLayer) {
        this.previousLayer = previousLayer;
        if(previousLayer!=null) {
            if(this.numberOfInputs!=previousLayer.numberOfNeuronsInLayer)throw new RuntimeException("Illegal previous layer");
            setInput(new ArrayList<>(previousLayer.numberOfNeuronsInLayer));
        }
    }

    public NeuralLayer getNextLayer() {
        return nextLayer;
    }

    void setNextLayer(NeuralLayer nextLayer) {

        this.nextLayer = nextLayer;
        if(nextLayer!=null) setOutput(new ArrayList<>(nextLayer.numberOfNeuronsInLayer));
        else setOutput(new ArrayList<>(this.numberOfNeuronsInLayer));
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

    private void setOutput(ArrayList<Double> output) {
        this.output = output;
    }
}
