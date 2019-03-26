package ActivationFunctions;

public class Sigmoid implements IActivationFunction{
    private double a=1.0;

    public Sigmoid(double a){
        this.a=a;
    }

    @Override
    public Double calc(Double outputBeforeActivation) {
        return 1.0/(1.0+Math.exp(-a*outputBeforeActivation));
    }
}
