import java.util.Arrays;

public class Main {


    public static void main(String[] args){

        Data data=new Data(Main.class.getResource("dataset.csv"));
        data.print();
        NeuralNetwork NN=new NeuralNetwork(1,1000,data);
        NN.runTnTby2cv(5);
    }
}

