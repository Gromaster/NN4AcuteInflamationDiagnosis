package ActivationFunctions;

public class Sigmoid {


    public Double calc(Double outputBeforeActivation) {
        return 1/(1+Math.exp(-outputBeforeActivation));
    }
}
