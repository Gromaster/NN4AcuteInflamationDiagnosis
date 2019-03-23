import java.util.Arrays;

public class NeuralNetwork {
    private int nodesNumber, layersNumber;
    private double[][] weights,biases;
    private Data data;

    public NeuralNetwork(int layersNumber, int nodesNumber, Data data) {
        this.setLayersNumber(layersNumber);
        this.setNodesNumber(nodesNumber);
        this.data   = data;
        weights     = np.random(getLayersNumber(),getNodesNumber());
        biases      = new double[getLayersNumber()][getNodesNumber()];

    }

    public void learn(int iterations){
        if(data==null) {
            throw new EmptyDataException("No data given to the NN");
        }
        double alfa=  0.01;
        for(int i=0;i<iterations;i++) {
            //Forward Propagation
            double[][] Results          = np.add(np.dot(weights, data.getPatientData()), biases);
            double[][] ActivationFunc   = np.sigmoid(Results);
            double cost                 = np.cross_entropy(getLayersNumber(), data.getPatientDiagnoses(), ActivationFunc);

            //Back Propagation
            double[][] dResults = np.subtract(ActivationFunc,data.getPatientDiagnoses());
            double[][] dWeights = np.divide(np.dot(dResults,np.T(data.getPatientData())),getLayersNumber());
            double[][] dBiases  = np.divide(dResults,getLayersNumber());

            //Gradient descent
            weights =np.subtract(weights,np.multiply(alfa, dWeights));
            biases  =np.subtract(biases,np.multiply(alfa,dBiases));
            if(i%(iterations/10)==0){
                System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~");
                System.out.println("Cost = "+cost);
                System.out.println("Predictions = " +Arrays.deepToString(ActivationFunc));
            }
        }
    }

    private int getLayersNumber() {
        return layersNumber;
    }

    private void setLayersNumber(int layersNumber) {
        this.layersNumber = layersNumber;
    }

    private int getNodesNumber() {
        return nodesNumber;
    }

    private void setNodesNumber(int nodesNumber) {
        this.nodesNumber = nodesNumber;
    }

    private class EmptyDataException extends RuntimeException {
        EmptyDataException(String s) {
            super(s);
            this.printStackTrace();
        }
    }
}
