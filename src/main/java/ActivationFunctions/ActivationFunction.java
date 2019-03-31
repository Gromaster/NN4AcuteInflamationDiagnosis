package ActivationFunctions;

public enum ActivationFunction implements IActivationFunction{
    STEP{
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation>0 ? 1.0 : 0.0;
        }

    },
    LINEAR{
        @Override
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation;
        }
    },
    SIGMOID{
        @Override
        public Double calc(Double outputBeforeActivation) {
            return 1/(1+Math.exp(-outputBeforeActivation));
        }
    },
    RELU{
        @Override
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation>0 ? outputBeforeActivation : 0.0;
        }
    }
}
