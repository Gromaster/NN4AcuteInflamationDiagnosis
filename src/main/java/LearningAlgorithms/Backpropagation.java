package LearningAlgorithms;

import Data.NeuralDataSet;
import NeuralNetworkPackage.NeuralLayer;
import NeuralNetworkPackage.NeuralNetwork;
import NeuralNetworkPackage.Neuron;

import java.util.ArrayList;

public class Backpropagation extends LearningAlgorithm {


    private ArrayList<ArrayList<Double>> error;
    private ArrayList<Double> generalError;
    private ArrayList<Double> overallError;
    private double overallGeneralError;

    public ArrayList<ArrayList<Double>> testingError;
    public ArrayList<Double> testingGeneralError;
    public ArrayList<Double> testingOverallError;
    public double testingOverallGeneralError;


    public double degreeGeneralError=2.0;
    public double degreeOverallError=0.0;

    public enum ErrorMeasurement {SimpleError, SquareError,NDegreeError,MSE}

    public ErrorMeasurement generalErrorMeasurement= ErrorMeasurement.SquareError;
    public ErrorMeasurement overallErrorMeasurement= ErrorMeasurement.MSE;

    private int currentRecord=0;

    private ArrayList<ArrayList<ArrayList<Double>>> newWeights;

    private double MomentumRate=0.7;
    private ArrayList<ArrayList<Double>> deltaNeuron;
    private ArrayList<ArrayList<ArrayList<Double>>> previousDeltaWeights;

    public Backpropagation(NeuralNetwork neuralNetwork){
        this.neuralNetwork=neuralNetwork;
        this.newWeights=new ArrayList<>();
        int numberOfHiddenLayers=this.neuralNetwork.getNumberOfHiddenLayers();
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int numberOfNeuronsInLayer,numberOfInputsInNeuron;
            this.newWeights.add(new ArrayList<>());
            if(l<numberOfHiddenLayers){
                numberOfNeuronsInLayer=this.neuralNetwork.getHiddenLayer(l)
                        .getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNetwork.getHiddenLayer(l)
                            .getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.newWeights.get(l).get(j).add(0.0);
                    }
                }
            }
            else{
                numberOfNeuronsInLayer=this.neuralNetwork.getOutputLayer()
                        .getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNetwork.getOutputLayer()
                            .getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.newWeights.get(l).get(j).add(0.0);
                    }
                }
            }
        }
    }

    public Backpropagation(NeuralNetwork neuralNetwork,NeuralDataSet trainDataSet, LearningMode learningMode){
        this(neuralNetwork);
        this.trainingDataSet=trainDataSet;
        this.generalError=new ArrayList<>();
        this.error=new ArrayList<>();
        this.overallError=new ArrayList<>();
        this.learningMode= learningMode;
        for(int i=0;i<trainDataSet.getNumberOfRecords();i++){
            this.generalError.add(null);
            this.error.add(new ArrayList<>());
            for(int j=0;j<neuralNetwork.getNumberOfOutputs();j++){
                if(i==0){
                    this.overallError.add(null);
                }
                this.error.get(i).add(null);
            }
        }
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
            else {
                numberOfNeuronsInLayer=neuralNetwork.getHiddenLayer(j).getNumberOfNeuronsInLayer();
            }
            for(int i=0;i<numberOfNeuronsInLayer;i++)
                deltaNeuron.get(j).add(null);
        }
    }

    private void initializeLastDeltaWeights() {
        this.previousDeltaWeights = new ArrayList<>();
        int numberOfHiddenLayers = neuralNetwork.getNumberOfHiddenLayers();
        for(int k=0;k<=numberOfHiddenLayers;k++){
            int numberOfNeuronsInLayer,numberOfInputsInNeuron;
            previousDeltaWeights.add(new ArrayList<>());
            if(k==numberOfHiddenLayers){
                numberOfNeuronsInLayer=neuralNetwork.getOutputLayer().getNumberOfNeuronsInLayer();
                numberOfInputsInNeuron=neuralNetwork.getOutputLayer().getNumberOfInputs();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    previousDeltaWeights.get(k).add(new ArrayList<>());
                    for(int i=0;i<numberOfInputsInNeuron;i++)
                        previousDeltaWeights.get(k).get(j).add(0.0);
                }

            }
            else {
                numberOfNeuronsInLayer = neuralNetwork.getHiddenLayer(k).getNumberOfNeuronsInLayer();
                numberOfInputsInNeuron = neuralNetwork.getHiddenLayer(k).getNumberOfInputs();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    previousDeltaWeights.get(k).add(new ArrayList<>());
                    for(int i=0;i<numberOfInputsInNeuron;i++)
                        previousDeltaWeights.get(k).get(j).add(0.0);
                }
            }
        }
    }




    @Override
    public void train(){
        epoch=0;
        int k=0;
        currentRecord=0;
        forward();
        forward(k);
        while (epoch< MAXEpoch && overallGeneralError>MinOverallError) {
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

    @Override
    public void forward(){
        for(int i=0;i<trainingDataSet.getNumberOfRecords();i++){
            neuralNetwork.setInputs(trainingDataSet.getInputRecord(i));
            neuralNetwork.calc();
            trainingDataSet.setNeuralOutput(i, neuralNetwork.getOutputs());
            generalError.set(i, generalError(trainingDataSet.getExpectedOutputRecord(i),trainingDataSet.getNeuralOutputRecord(i)));
            for(int j=0;j<neuralNetwork.getNumberOfOutputs();j++){
                error.get(i).set(j, simpleError(trainingDataSet.getExpectedOutputRecord(i).get(j), trainingDataSet.getNeuralOutputRecord(i).get(j)));
            }
        }
        for(int j=0;j<neuralNetwork.getNumberOfOutputs();j++){
            overallError.set(j, overallError(trainingDataSet.getIthExpectedOutput(j), trainingDataSet.getIthNeuralOutput(j)));
        }
        overallGeneralError=overallGeneralError(trainingDataSet.getExpectedOutputData(),trainingDataSet.getNeuralOutputData());
    }

    @Override
    public void forward(int i){
        neuralNetwork.setInputs(trainingDataSet.getInputRecord(i));
        neuralNetwork.calc();
        trainingDataSet.setNeuralOutput(i, neuralNetwork.getOutputs());
        generalError.set(i,generalError(trainingDataSet.getExpectedOutputRecord(i),trainingDataSet.getNeuralOutputRecord(i)));
        for(int j=0;j<neuralNetwork.getNumberOfOutputs();j++){
            overallError.set(j,overallError(trainingDataSet.getIthExpectedOutput(j), trainingDataSet.getNeuralOutputRecord(i)));
            error.get(i).set(j,simpleError(trainingDataSet.getIthExpectedOutput(j).get(i), trainingDataSet.getIthNeuralOutput(j).get(i)));
        }
        overallGeneralError=overallGeneralError(trainingDataSet.getExpectedOutputData(),trainingDataSet.getNeuralOutputData());

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
                            currNewWeight= neuralNetwork.getOutputLayer().getWeight(i,j);
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
            currLayer=neuralNetwork.getHiddenLayer(layer);
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

    @Override
    public Double calcNewWeight(int layer, int input, int neuron){
        Double deltaWeight=calcDeltaWeight(layer,input,neuron);
        return newWeights.get(layer).get(neuron).get(input)+deltaWeight;
    }

    @Override
    public Double calcNewWeight(int layer, int input, int neuron, double error){
        return null;
    }

    private Double generalError(ArrayList<Double> outputExpected,ArrayList<Double> neuralOutput){
        int size=outputExpected.size();
        Double result=0.0;
        for(int i=0;i<size;i++){
            result+=Math.pow(outputExpected.get(i)-neuralOutput.get(i),degreeGeneralError);
        }
        if(generalErrorMeasurement==ErrorMeasurement.MSE)
            result*=(1.0/size);
        else
            result*=(1.0/degreeGeneralError);
        return result;
    }

    private Double overallError(ArrayList<Double> outputExpected,ArrayList<Double> neuralOutput){
        int size=outputExpected.size();
        Double result=0.0;
        for(int i=0;i<size;i++)
            result+=Math.pow(outputExpected.get(i)-neuralOutput.get(i),degreeOverallError);
        if(overallErrorMeasurement==ErrorMeasurement.MSE)
            result*=(1.0/size);
        else
            result*=(1.0/degreeOverallError);
        return result;
    }

    private Double simpleError(Double outputExpected,Double neuralOutput){ return outputExpected-neuralOutput;}

    private Double squareError(Double outputExpected,Double neuralOutput){ return Math.pow(outputExpected-neuralOutput,2.0);}

    private Double overallGeneralError(ArrayList<ArrayList<Double>> expectedOutput,ArrayList<ArrayList<Double>> neuralOutput) {
        int N=expectedOutput.size();
        int M=expectedOutput.get(0).size();
        Double result=0.0;
        for(int i=0;i<N;i++){
            Double partialResult = 0.0;
            for(int j=0;j<M;j++){
                partialResult+=Math.pow(expectedOutput.get(i).get(j)-neuralOutput.get(i).get(j),degreeGeneralError);
            }
            if(generalErrorMeasurement==ErrorMeasurement.MSE)
                result+=Math.pow((1.0/M)*partialResult,degreeOverallError);
            else
                result+=Math.pow((1.0/degreeGeneralError)*partialResult,degreeOverallError);
        }
        return (1.0/N)*result;
    }




}
