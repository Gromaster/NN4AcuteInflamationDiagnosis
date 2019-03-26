package NeuralNetworkPackage;

public class InputLayer extends NeuralLayer{
    public InputLayer(int numberOfInputs){
        this.numberOfInputs=numberOfInputs;
        setPreviousLayer(null);
    }

}
