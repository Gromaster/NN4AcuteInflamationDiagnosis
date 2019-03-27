package NeuralNetworkPackage;

import ActivationFunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class InputLayer extends NeuralLayer{
     InputLayer(int numberOfInputs){
        super(numberOfInputs,numberOfInputs,ActivationFunction.LINEAR);
        setPreviousLayer(null);
        init();
    }

    protected void init(){
        for (int i=0;i<numberOfNeuronsInLayer;i++) {
            try {
                neurons.get(i).setActivationFunction(activationFunction);
                neurons.get(i).setWeights(new ArrayList<>(Collections.nCopies(numberOfInputs+1,0.0)));
                neurons.get(i).getWeights().set(i,1.0);

                System.out.println(neurons.get(i).getWeights().toString());
            }
            catch(IndexOutOfBoundsException | NullPointerException e) {
                neurons.add(new Neuron(numberOfInputs,activationFunction));
                neurons.get(i).setWeights(new ArrayList<>(Collections.nCopies(numberOfInputs+1,0.0)));
                neurons.get(i).getWeights().set(i,1.0);
                System.out.println(neurons.get(i).getWeights().toString()+" w catch");
            }
        }
    }

}
