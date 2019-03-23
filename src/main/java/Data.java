public class Data {
    private final double[][] patientData;
    private final double[][] patientDiagnoses;

    public Data(double[][] patientData, double[][] patientDiagnoses) {
        this.patientData = patientData;
        this.patientDiagnoses = patientDiagnoses;
    }

    double[][] getPatientData() {
        return patientData;
    }


    double[][] getPatientDiagnoses() {
        return patientDiagnoses;
    }

}
