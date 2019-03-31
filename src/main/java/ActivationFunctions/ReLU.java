package ActivationFunctions;

public class ReLU {


    public Double calc(Double outputBeforeActivation) {
        return outputBeforeActivation>0 ? outputBeforeActivation : 0.0;
    }
}
