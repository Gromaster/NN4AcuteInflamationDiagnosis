

public class Main {


    public static void main(String[] args){
        Data data=new Data(Main.class.getResource("dataset.csv"));
        data.print();
        NeuralNetwork NN=new NeuralNetwork(1,100,data);
        NN.learn(100);
    }
}
