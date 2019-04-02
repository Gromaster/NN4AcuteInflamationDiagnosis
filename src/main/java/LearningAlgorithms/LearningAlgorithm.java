package LearningAlgorithms;

import Data.NeuralDataSet;
import NeuralNetworkPackage.NeuralNetwork;

import java.util.ArrayList;

public abstract class LearningAlgorithm {
    protected NeuralNetwork neuralNetwork; //the one trained by this algorithm
    public enum LearningMode{ONLINE,BATCH};
    protected enum LearningParagigm {SUPERVISED,UNSUPERVISED};
    protected LearningMode learningMode;
    protected final int MAXEpoch =10000;
    protected int epoch=0;
    protected double MinOverallError=0.001; //Error causing neural network to stop learning
    protected double LearningRate=0.1;
    protected NeuralDataSet trainingDataSet;
    protected NeuralDataSet testingDataSet;
    protected NeuralDataSet validatingDataSet;
    public boolean printTraining=false;
    public abstract void train() throws NeuralException;
    public abstract void forward() throws NeuralException;
    public abstract void forward(int k) throws NeuralException;//process neural network with kth input data record
    public abstract Double calcNewWeight(int layer,int input,int neuron) throws NeuralException;
    public abstract Double calcNewWeight(int layer,int input, int neuron,double error)throws NeuralException;//weight update providing given error

    private class NeuralException extends Exception{}
}
