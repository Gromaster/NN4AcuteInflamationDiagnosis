package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    protected ArrayList<Double> weights;
    private ArrayList<Double> input;
    private Double output;
    private Double outputBeforeActivation;
    protected Double bias=1.0;
    private int numberOfInputs=0;
    private IActivationFunction activationFunction;


    public Neuron(int numberOfInputs, IActivationFunction activationFunction){
        this.numberOfInputs=numberOfInputs;
        weights =new ArrayList<>(numberOfInputs+1);
        input=new ArrayList<>(numberOfInputs);
        this.activationFunction=activationFunction;
    }

    public void init(){
        Random r=new Random();
        for(int i=0;i<numberOfInputs;i++){
            this.weights.set(i,r.nextDouble());
        }
    }

    public void calc(){
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

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public void setWeights(ArrayList<Double> weights) {
        this.weights = weights;
    }

    public ArrayList<Double> getInput() {
        return input;
    }

    public void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    public Double getOutput() {
        return output;
    }

    public void setOutput(Double output) {
        this.output = output;
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

    public void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
}
