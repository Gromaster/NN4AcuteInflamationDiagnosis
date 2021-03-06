package Data;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NeuralDataSet {
    private ArrayList<ArrayList<Double>> patientData;
    private ArrayList<ArrayList<Double>> patientDiagnoses;
    private ArrayList<ArrayList<Double>> neuralOutput;
    private int numberOfRecords=0;
    private int numberOfDiagnoses=0;
    private int numberOfParameters=0;


    public NeuralDataSet(String filePath) {
        dataAcquisition(filePath);
    }

    private void dataAcquisition(String filePath) {
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(new File(filePath)));
            String line;
            ArrayList<String[]> buffer=new ArrayList<>();
            while((line=br.readLine())!=null){
                buffer.add(line.split(";"));
            }
            int numberOfRecords=buffer.size();
            setNumberOfRecords(buffer.size());
            numberOfDiagnoses   = 2;
            numberOfParameters  = buffer.get(0).length-numberOfDiagnoses;
            patientData         = new ArrayList<>(buffer.size());
            patientDiagnoses    = new ArrayList<>(buffer.size());

            for(int i=0;i<numberOfRecords;i++){
                int j=0;
                patientData.add(new ArrayList<>(Collections.nCopies(numberOfParameters,0.0)));
                patientDiagnoses.add(new ArrayList<>(Collections.nCopies(numberOfDiagnoses,0.0)));
                for(;j<numberOfParameters;j++) {
                    patientData.get(i).set(j,checkDataElement(buffer.get(i)[j]));
                }
                for(int k=0;j<numberOfParameters+numberOfDiagnoses;j++,k++)
                    patientDiagnoses.get(i).set(k,checkDataElement(buffer.get(i)[j]));
            }
            this.numberOfRecords=numberOfRecords;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double checkDataElement(String str)
    {
        if(str.contains("yes"))return 1;
        else if(str.contains("no"))return 0;
        else if(35<=Double.parseDouble(str) && Double.parseDouble(str)<=42) return (Double.parseDouble(str)-35)/7;
        else throw new DataReadingException("Not able to find right value for data:\""+str+"\"");
    }

     void print(){
        System.out.printf("%25s\t%20s\t%20s\t%20s\t%20s\t%20s\t%20s\t%20s\n","Temperature of patient","Occurrence of nausea","Lumbar pain","Urine pushing","Micturition pains","Burning of urethra","dec: Inflammation","dec: Nephritis");
        for(int i=0;i<numberOfRecords;i++) {
            System.out.printf("%-5d",i);
            for (int j = 0; j < patientData.get(i).size(); j++)
                System.out.printf("%20f\t", patientData.get(i).get(j));
            for (int j = 0; j < patientDiagnoses.get(i).size();j++)
                System.out.printf("%20f\t",patientDiagnoses.get(i).get(j));
            System.out.println();
        }
    }

    public ArrayList<ArrayList<Double>> getArrayInputData() {
        ArrayList<ArrayList<Double>> returnList=new ArrayList<>(patientData.size());
        for(ArrayList<Double> singleRecord:patientData)
            returnList.add(new ArrayList<>(singleRecord));
        return returnList;
    }


    public int getNumberOfRecords() {
         return numberOfRecords;
     }

     private void setNumberOfRecords(int numberOfRecords) {
         this.numberOfRecords = numberOfRecords;
     }

    public ArrayList<Double> getIthInputArrayList(int i) {
        return new ArrayList<>(this.patientData.get(i));
    }

    public ArrayList<Double> getInputRecord(int i) {
        return new ArrayList<>(patientData.get(i));
    }

    public void setNeuralOutput(int i, ArrayList<Double> outputs) {
        if(this.neuralOutput==null || outputs.size()!=getNumberOfRecords())
            this.neuralOutput=new ArrayList<>(getNumberOfRecords());
        this.neuralOutput.set(i, new ArrayList<>(outputs));
    }

    public ArrayList<ArrayList<Double>> getExpectedOutputData() {
        ArrayList<ArrayList<Double>> result=new ArrayList<>(getNumberOfRecords());
        for(ArrayList<Double> array:patientDiagnoses)
            result.add(new ArrayList<>(array));
        return result;
    }

    public ArrayList<Double> getIthExpectedOutput(int i) {//return specified Ith column of expected output
        ArrayList<Double> result=new ArrayList<>(getNumberOfRecords());
        for(ArrayList<Double> array: patientDiagnoses)
            result.add(array.get(i));
        return result;
    }

    public ArrayList<Double> getExpectedOutputRecord(int i) {
        return new ArrayList<>(patientDiagnoses.get(i));
    }

    public ArrayList<Double> getNeuralOutputRecord(int i) {
        return new ArrayList<>(neuralOutput.get(i));
    }

    public ArrayList<Double> getIthNeuralOutput(int i) {
        ArrayList<Double> result=new ArrayList<>(getNumberOfRecords());
        for(ArrayList<Double> array:neuralOutput)
            result.add(array.get(i));
        return result;
    }

    public ArrayList<ArrayList<Double>> getNeuralOutputData() {
        ArrayList<ArrayList<Double>> result=new ArrayList<>(getNumberOfRecords());
        for(ArrayList<Double> array:neuralOutput)
            result.add(new ArrayList<>(array));
        return result;
    }

    class DataReadingException extends RuntimeException{
        DataReadingException(String s){
            super(s);
        }
    }
 }
