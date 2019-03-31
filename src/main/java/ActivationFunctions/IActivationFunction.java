package ActivationFunctions;

public interface IActivationFunction {
    Double calc(Double outputBeforeActivation);
    Double derivative();
}
