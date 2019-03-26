package ActivationFunctions;

public interface IActivationFunction {
    Double calc(Double outputBeforeActivation);
    public enum ActivationFunctionENUM{
        STEP, LINEAR, SIGMOID, HYPERTAN, RELU
    }
}
