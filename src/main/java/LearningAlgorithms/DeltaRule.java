package LearningAlgorithms;

import Data.NeuralDataSet;
import NeuralNetworkPackage.*;

import java.util.ArrayList;

public class DeltaRule extends LearningAlgorithm {
    public ArrayList<ArrayList<Double>> error; //errors for each output record
    public ArrayList<Double> generalError;
    public ArrayList<Double> overallError;
    public double overallGeneralError; //cost function result
    public double degreeGeneralError=2.0;
    public double degreeOverallError=0.0;
    public enum ErrorMeasurement {SimpleError, SquareError, NDegreeError, MSE}
    public ErrorMeasurement generalErrorMeasurement=ErrorMeasurement.SquareError;
    public ErrorMeasurement overallErrorMeasurement=ErrorMeasurement.MSE;
    protected int currentRecord=0;
    protected ArrayList<ArrayList<ArrayList<Double>>> newWeights;
    protected LearningMode learningMode;

    public DeltaRule(NeuralNetwork neuralNetwork,NeuralDataSet trainDataSet,LearningMode learningMode){
        this.neuralNetwork=neuralNetwork;
        this.trainingDataSet=trainDataSet;
        this.learningMode=learningMode;
    }

    @Override
    public void train() throws NeuralException {
        switch (learningMode) {
            case BATCH:
                epoch = 0;
                forward();
                while (epoch < MAXepoch && overallGeneralError > MinOverallError) {
                    epoch++;
                    for (int j = 0; j < neuralNetwork.getNumberOfOutputs(); j++) {
                        for (int i = 0; i <= neuralNetwork.getNumberOfInputs(); i++) {
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        }
                    }
                    applyNewWeights();
                    forward();
                }
                break;
            case ONLINE:
                epoch = 0;
                int k = 0;
                currentRecord = 0;
                forward(k);
                while (epoch < MAXepoch && overallGeneralError > MinOverallError) {
                    for (int j = 0; j < neuralNetwork.getNumberOfOutputs(); j++) {
                        for (int i = 0; i < neuralNetwork.getNumberOfInputs();i++) {
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        }
                    }
                    applyNewWeights();
                    currentRecord = ++k;
                    if (k >= trainingDataSet.getNumberOfRecords()) {
                        k = 0;
                        currentRecord = 0;
                        epoch++;
                    }
                    forward(k);
                }
                break;
        }
    }

    @Override
    public void forward() throws NeuralException {

    }

    @Override
    public void forward(int k) throws NeuralException {

    }

    @Override
    public Double calcNewWeight(int layer, int input, int neuron) throws NeuralException {
        Double deltaWeight = LearningRate;
        Neuron currNeuron = neuralNetwork.getOutputLayer().getNeuron(neuron);
        switch (learningMode){
            case BATCH:
                ArrayList<Double> derivativeResult=currNeuron.derivativeBatch(trainingDataSet.getArrayInputData());
                ArrayList<Double> _ithInput;
                if(input<currNeuron.getNumberOfInputs()){
                    _ithInput=trainingDataSet.getIthInputeArrayList(input);
                }
                else {
                    _ithInput=new ArrayList<>();
                    for(int i=0;i<trainingDataSet.getNumberOfRecords();i++){
                        _ithInput.add(1.0);
                    }
                }
                Double multDerivResultIthInput=0.0;
                for(int i=0;i<trainingDataSet.getNumberOfRecords();i++){
                    multDerivResultIthInput+=error.get(i).get(neuron)*derivativeResult.get(i)*_ithInput.get(i);
                }
                deltaWeight*=multDerivResultIthInput;
                break;
            case ONLINE:
                deltaWeight*=error.get(currentRecord).get(neuron);
                deltaWeight*=currNeuron.derivative(neuralNetwork.getInput());
                if(input<currNeuron.getNumberOfInputs()){
                    deltaWeight*=neuralNetwork.getInput(input);
                }
                break;
        }
        return currNeuron.getWeight(input)+deltaWeight;
    }

    @Override
    public Double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException {

        return null;
    }

    class NeuralException extends Exception {
    }
}
