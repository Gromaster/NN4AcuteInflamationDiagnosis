package NeuralNetworkPackage;

import ActivationFunctions.IActivationFunction;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    private ArrayList<Double> weights;
    private ArrayList<Double> inputs;
    private Double output=0.0;
    private Double outputBeforeActivation;
    private Double bias = 1.0;
    private int numberOfInputs;
    private IActivationFunction activationFunction;
    private int i = 0;


    public Neuron(Neuron original){
        this.setWeights(original.getWeights());
        this.setInputs(original.getInputs());
        this.setOutput(original.getOutput());
        this.setOutputBeforeActivation(original.getOutputBeforeActivation());
        this.setBias(original.getBias());
        this.setNumberOfInputs(original.getNumberOfInputs());
        this.setActivationFunction(original.getActivationFunction());

    }

    public Neuron(int numberOfInputs, IActivationFunction activationFunction) {
        this.numberOfInputs = numberOfInputs;
        weights = new ArrayList<>((numberOfInputs+1));
        inputs = new ArrayList<>(numberOfInputs);
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
            if (inputs != null && weights != null) {
                for (int i = 0; i <= numberOfInputs; i++) {
                    outputBeforeActivation += (i == numberOfInputs ? bias : inputs.get(i)) * weights.get(i);
                }
            }
        }
        output=activationFunction.calc(outputBeforeActivation);
    }

    public Double derivative(ArrayList<Double> input){
        Double outputBeforeActivation=0.0;
        if(numberOfInputs>0){
            if(weights!=null){
                for(int i=0;i<=numberOfInputs;i++){
                    outputBeforeActivation+=(i==numberOfInputs?bias:input.get(i))*weights.get(i);
                }
            }
        }
        return activationFunction.derivative(outputBeforeActivation);
    }



    public double getWeight(int i){
        return weights.get(i);
    }

    public ArrayList<Double> getWeights() {
        return new ArrayList<>(weights);
    }

    public void setWeights(ArrayList<Double> weights) {
        this.weights = new ArrayList<>(weights);
    }

    public ArrayList<Double> getInputs() {
        return new ArrayList<>(inputs);
    }

    public void setInputs(ArrayList<Double> inputs) {
        this.inputs = new ArrayList<>(inputs);
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

     void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public String toString(){
        return "Neuron: "+ "Input:"+this.inputs +" weights: "+weights.toString()+"\n";
    }

    public Double getInput(int i) {
        return inputs.get(i);
    }

    public void setWeight(int i, double newWeight) {
        this.weights.set(i,newWeight);
    }
}
