package LearningAlgorithms;

import Data.NeuralDataSet;
import NeuralNetworkPackage.NeuralLayer;
import NeuralNetworkPackage.NeuralNetwork;
import NeuralNetworkPackage.Neuron;

import java.util.ArrayList;

public class Backpropagation extends DeltaRule {
    private double MomentumRate=0.7;
    public ArrayList<ArrayList<Double>> deltaNeuron;
    public ArrayList<ArrayList<ArrayList<Double>>> lastDeltaWeights;


    public Backpropagation(NeuralNetwork neuralNetwork, NeuralDataSet trainDataSet, DeltaRule.LearningMode learningMode) {
        super(neuralNetwork,trainDataSet,learningMode);
        initializeDeltaNeuron();
        initializeLastDeltaWeights();
    }

    private void initializeDeltaNeuron() {
        deltaNeuron=new ArrayList<>();
        int numberOfHiddenLayers=neuralNetwork.getNumberOfHiddenLayers();
        for(int j=0;j<=numberOfHiddenLayers;j++){
            int numberOfNeuronsInLayer;
            deltaNeuron.add(new ArrayList<Double>());
            if(j==numberOfHiddenLayers){
                numberOfNeuronsInLayer=neuralNetwork.getOutputLayer().getNumberOfNeuronsInLayer();
            }
        }
    }

    @Override
    public void train() throws NeuralException {
        epoch=0;
        int k=0;
        currentRecord=0;
        forward();
        forward(k);
        while (epoch<MAXepoch && overallGeneralError>MinOverallError) {
            backward();
            switch (learningMode) {
                case BATCH:
                    if(k==trainingDataSet.getNumberOfRecords()-1)
                        applyNewWeights();
                    break;
                case ONLINE:
                    applyNewWeights();
            }
            currentRecord=++k;
            if(k>=trainingDataSet.getNumberOfRecords()){
                epoch++;
                currentRecord=0;
                k=0;
            }
            forward(k);
        }
    }

    public void backward(){
        int numberOfLayers=neuralNetwork.getNumberOfHiddenLayers();
        for(int l=numberOfLayers;l>=0;l--){
            int numberOfNeuronsInLayer=deltaNeuron.get(l).size();
            for(int j=0;j<numberOfNeuronsInLayer;j++){
                for(int i=0;i<newWeights.get(l).get(j).size();i++){
                    double currNewWeight=this.newWeights.get(l).get(j).get(i);
                    if(currNewWeight==0.0 && epoch==0)
                        if(l==numberOfLayers)
                            currNewWeight= (double) neuralNetwork.getOutputLayer().getWeight(i,j);
                        else
                            currNewWeight=neuralNetwork.getHiddenLayer(l).getWeight(i,j);
                        double deltaWeight=calcDeltaWeight(l,i,j);
                        newWeights.get(l).get(j).set(i,currNewWeight+deltaWeight);
                }
            }
        }
    }

    public Double calcDeltaWeight(int layer,int input,int neuron){
        Double deltaWeight=1.0;
        NeuralLayer currLayer;
        Neuron currNeuron;
        double currDeltaNeuron;
        if(layer==neuralNetwork.getNumberOfHiddenLayers()){
            currLayer=neuralNetwork.getOutputLayer();
            currNeuron=currLayer.getNeuron(neuron);
            currDeltaNeuron=error.get(neuron)*currNeuron.derivative(currLayer.getInputs());
        }
        else {
            currLayer=neuralNetwork.getHiddenLayers(layer);
            currNeuron=currLayer.getNeuron(neuron);
            double sumDeltaNextLayer=0;
            NeuralLayer nextLayer=currLayer.getNextLayer();
            for(int k=0;k<nextLayer.getNumberOfNeuronsInLayer();k++)
                sumDeltaNextLayer+=nextLayer.getWeight(neuron,k)*deltaNeuron.get(layer+1).get(k);
            currDeltaNeuron=sumDeltaNextLayer*currNeuron.derivative(currLayer.getInputs());
        }
        deltaNeuron.get(layer).set(neuron,currDeltaNeuron);
        deltaWeight*=currDeltaNeuron;
        if(input<currNeuron.getNumberOfInputs()){
            deltaWeight*=currNeuron.getInput(input);
        }
        return deltaWeight;
    }






}
