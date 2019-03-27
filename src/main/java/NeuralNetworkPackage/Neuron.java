package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    private ArrayList<Double> weights;
    private ArrayList<Double> input;
    private Double output=0.0;
    private Double outputBeforeActivation;
    private Double bias = 1.0;
    private int numberOfInputs;
    private IActivationFunction activationFunction;
    private int i = 0;


    Neuron(int numberOfInputs, IActivationFunction activationFunction) {
        this.numberOfInputs = numberOfInputs;
        weights = new ArrayList<>((numberOfInputs+1));
        input = new ArrayList<>(numberOfInputs);
        this.activationFunction = activationFunction;
        init();
    }

     void init(){
        Random r=new Random();
        for(int i=0;i<numberOfInputs+1;i++){
            try{
                this.weights.set(i,r.nextDouble());
            }
            catch(IndexOutOfBoundsException e)
            {
                weights.add(r.nextDouble());
            }
        }
    }
     void calc(){
        outputBeforeActivation=0.0;
        if(numberOfInputs>0) {
            if (input != null && weights != null) {
                for (int i = 0; i <= numberOfInputs; i++) {
                    outputBeforeActivation += (i == numberOfInputs ? bias : input.get(i)) * weights.get(i);
                }
            }
        }
        output=activationFunction.calc(outputBeforeActivation);
    }

     ArrayList<Double> getWeights() {
        return weights;
    }

     void setWeights(ArrayList<Double> weights) {
        this.weights = weights;
    }

    public ArrayList<Double> getInput() {
        return input;
    }

     void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    Double getOutput() {
        return output;
    }

    public Double getOutputBeforeActivation() {
        return outputBeforeActivation;
    }

    public void setOutputBeforeActivation(Double outputBeforeActivation) {
        this.outputBeforeActivation = outputBeforeActivation;
    }

    public Double getBias() {
        return bias;
    }

    public void setBias(Double bias) {
        this.bias = bias;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }

    public IActivationFunction getActivationFunction() {
        return activationFunction;
    }

     void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    public String toString(){
        return "Neuron: "+ "Input:"+this.input +" weights: "+weights.toString()+"\n";
    }
}
