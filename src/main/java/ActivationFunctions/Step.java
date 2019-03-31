package ActivationFunctions;

public class Step  {



    public Double calc(Double outputBeforeActivation) {
        return outputBeforeActivation>0 ? 1.0 : 0.0;
    }
}
