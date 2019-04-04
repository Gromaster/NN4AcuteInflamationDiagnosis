package ActivationFunctions;

public enum ActivationFunction implements IActivationFunction{
    STEP{
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation>0 ? 1.0 : 0.0;
        }

        @Override
        public Double derivative(Double outputBeforeActivation) {
            return outputBeforeActivation!=0.0 ? 0.0 : Double.MAX_VALUE;
        }

    },
    LINEAR{
        private Double slopeFactor=1.0;

        public Double getSlopeFactor() {
            return slopeFactor;
        }

        public void setSlopeFactor(Double slopeFactor) {
            this.slopeFactor = slopeFactor;
        }

        @Override
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation*slopeFactor;
        }

        @Override
        public Double derivative(Double outputBeforeActivation) {
            return slopeFactor;
        }
    },
    SIGMOID{
        @Override
        public Double calc(Double outputBeforeActivation) {
            return 1/(1+Math.exp(-outputBeforeActivation));
        }

        @Override
        public Double derivative(Double outputBeforeActivation) {
            return calc(outputBeforeActivation)*(1-calc(outputBeforeActivation));
        }
    },
    RELU{
        private Double slopeFactor=1.0;

        @Override
        public Double calc(Double outputBeforeActivation) {
            return outputBeforeActivation>0 ? outputBeforeActivation*slopeFactor : 0.0;
        }

        @Override
        public Double derivative(Double outputBeforeActivation) {
            return outputBeforeActivation>0 ? slopeFactor : 0.0;
        }

        public Double getSlopeFactor() {
            return slopeFactor;
        }

        public void setSlopeFactor(Double slopeFactor) {
            this.slopeFactor = slopeFactor;
        }
    }
}
