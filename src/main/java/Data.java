import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;

 class Data {
    private double[][] patientData;
    private double[][] patientDiagnoses;

    Data(URL filePath) {
        dataAcquisition(filePath);
    }

    private void dataAcquisition(URL filePath) {
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(new File(filePath.toURI())));
            String line;
            ArrayList<String[]> buffor=new ArrayList<>();
            while((line=br.readLine())!=null){
                buffor.add(line.split(";"));
            }
            patientData=new double[buffor.size()][buffor.get(0).length-2];
            patientDiagnoses=new double[buffor.size()][2];
            for(int i=0;i<buffor.size();i++){
                int j=0;
                for(;j<buffor.get(i).length-2;j++)
                    patientData[i][j]=checkDataElement(buffor.get(i)[j]);
                for(int k=0;j<buffor.get(i).length;j++,k++)
                    patientDiagnoses[i][k]=checkDataElement(buffor.get(i)[j]);
            }
        } catch (IOException|URISyntaxException e) {
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

    double[][] getPatientData() {
        return patientData;
    }


    double[][] getPatientDiagnoses() {
        return patientDiagnoses;
    }

    void print(){
        System.out.printf("%20s\t%20s\t%20s\t%20s\t%20s\t%20s\t%20s\t%20s\n","Temperature of patient","Occurrence of nausea","Lumbar pain","Urine pushing","Micturition pains","Burning of urethra","dec: Inflammation","dec: Nephritis");
        for(int i=0;i<patientData.length;i++) {
            for (int j = 0; j < patientData[i].length; j++)
                System.out.printf("%20f\t", patientData[i][j]);
            for (int j = 0; j < patientDiagnoses[i].length;j++)
                System.out.printf("%20f\t",patientDiagnoses[i][j]);
            System.out.println();
        }
    }
    class DataReadingException extends RuntimeException{
        DataReadingException(String s){
            super(s);
        }
    }
}
