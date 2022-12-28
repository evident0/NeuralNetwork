public class Example {

    double x1;
    double x2;
    Category category;

    boolean isCorrect = false;

    public Example(double x1, double x2, Category category) {
        this.x1 = x1;
        this.x2 = x2;
        this.category = category;
    }

    public void setIsCorrect(boolean isCorrect) {
        this.isCorrect = isCorrect;
    }

    //used for writing to file
    public String toString() {
        return x1 + "," + x2 + "," + category + "," + isCorrect;
    }

}
