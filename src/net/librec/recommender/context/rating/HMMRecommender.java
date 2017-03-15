package net.librec.recommender.context.rating;

import net.librec.common.LibrecException;

import java.util.*;

import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.AbstractRecommender;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by jake on 3/4/2017.
 */
public class HMMRecommender extends AbstractRecommender {

    /**
     * number of states
     */
    private int numStates;

    /**
     * size of output vocabulary
     */
    private int sigmaSize;

    /**
     * size of output vocabulary
     */
    private int numSteps;

    /**
     * initial state probabilities
     */
    private double pi[];

    /**
     * transition probabilities
     */
    private double transProb[][];

    /**
     * emission probabilities
     */
    private double emissionProb[][];

    /**
     * user transition probabilities cache
     */
    private double userTransProb[][][];

    /**
     * user emission probabilities cache
     */
    private double userEmissionProb[][][];
    /**
     * user initial state probabilities cache
     */
    private double userPi[][];


    private int userSampleData[][];
    /**
     * User Artist Id lookups
     **/
    private List<SequenceRecord> userStateLookup;

    /**
     * User Artist Id lookups
     **/
    private HashMap<Integer, List<Integer>> userStateData;

    /**
     * matrix of time stamp
     */
    private static SparseMatrix timeMatrix;


    private List<UserItemStateCount> userItemStateCounts;


    @Override
    protected void setup() throws LibrecException {
        super.setup();
        numStates = conf.getInt("rec.hmm.numStates", 6);
        sigmaSize = conf.getInt("rec.hmm.sigmaSize", 12);
        numSteps = 100;//conf.getInt("rec.hmm.numSteps", 100);
        pi = new double[numStates];
        transProb = new double[numStates][numStates];
        emissionProb = new double[numStates][sigmaSize];


        userPi = new double[numUsers][numStates];
        userTransProb = new double[numUsers][numStates][numStates];
        userEmissionProb = new double[numUsers][numStates][sigmaSize];
        userSampleData = new int[numUsers][numStates];

        for (int s = 0; s < numStates; s++) {
            if (s == 0) {
                pi[s] = .25;
            } else
                pi[s] = .25;
        }

        for (int s = 0; s < numStates; s++) {
            for (int s1 = 0; s1 < numStates; s1++) {
                if (s == 0) {
                    transProb[s][s1] = .39;
                }
                if (s == 1) {
                    transProb[s][s1] = .21;
                }
                if (s == 2) {
                    transProb[s][s1] = .15;
                }
                if (s == 3) {
                    transProb[s][s1] = .25;
                }

            }
        }

        for (int i = 0; i < numStates; i++) {
            for (int s = 0; s < sigmaSize; s++) {
                if (i == 0)
                    emissionProb[i][s] = .40;

                if (i == 1 || i == 2)
                    emissionProb[i][s] = .10;

                if (i == 3)
                    emissionProb[i][s] = .20;


            }
        }


        timeMatrix = (SparseMatrix) getDataModel().getDatetimeDataSet();
        userStateLookup = new ArrayList<>();
        userStateData = new HashMap<Integer, List<Integer>>();
        userItemStateCounts = new ArrayList<UserItemStateCount>();


    }

    @Override
    protected void trainModel() throws LibrecException {

        //Define the time frames and find the user's most popular argument
        for (int u = 0; u < numUsers; u++) {
            List<Integer> userItems = trainMatrix.getColumns(u);
            List<Integer> oi = new ArrayList<Integer>();
            for (int i = 0; i < sigmaSize; i++) {
                int max = 0;
                int favArtistForTimeFrame = -1;
                for (int t = 0; t < timeMatrix.getRows(u).size(); t++) {
                    final int i1 = i;
                    final int u1 = u;
                    double h = timeMatrix.get(u, i) / 1000;
                    final int timeframe = (int) Math.round(h / 6);
                    if (userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getItem() == i1 && x.getState() == timeframe).findFirst().isPresent()) {
                        UserItemStateCount uisc = userItemStateCounts.stream().filter(x -> x.getUser() == u1
                                && x.getItem() == i1
                                && x.getState() == timeframe).findFirst().get();
                        uisc.iterateCount();
                        if (uisc.getCount() > max) {
                            max = uisc.getCount();
                            favArtistForTimeFrame = uisc.getItem();
                        }
                    } else {
                        UserItemStateCount uisc = new UserItemStateCount(u1, i1, timeframe, 1);
                        userItemStateCounts.add(uisc);
                        if (max == 0) {
                            max = uisc.getCount();
                        }
                    }
                    if (t + 1 == timeMatrix.getRows(u).size()) {
                        oi.add(favArtistForTimeFrame);
                    }
                }
            }

            //Do HMM Calculations
            int[] sampleData = new int[oi.size()];
            for (int i = 0; i < oi.size(); i++) {
                if (oi.get(i) != -1) {
                    sampleData[i] = oi.get(i);
                }
                userSampleData[u] = sampleData;
                int T = sampleData.length;
                double[][] fwd;
                double[][] bwd;

                double pi1[] = new double[numStates];
                double a1[][] = new double[numStates][numStates];
                double b1[][] = new double[numStates][sigmaSize];

                for (int s = 0; s < numSteps; s++) {
                    // calculation of Forward- und Backward Variables from the current model
                    fwd = forwardProc(sampleData);
                    bwd = backwardProc(sampleData);

                    // re-estimation of initial state probabilities
                    for (int k = 0; k < numStates; k++) {
                        pi1[k] = gamma(k, 0, sampleData, fwd, bwd);
                    }
                    // re-estimation of transition probabilities
                    for (int i1 = 0; i1 < numStates; i1++) {
                        for (int j = 0; j < numStates; j++) {
                            double num = 0;
                            double denom = 0;
                            for (int t = 0; t <= T - 1; t++) {
                                num += p(t, i1, j, sampleData, fwd, bwd);
                                denom += gamma(i1, t, sampleData, fwd, bwd);
                            }
                            a1[i1][j] = divide(num, denom);
                        }
                    }

                    //re-estimation of emission probabilities
                    for (int i2 = 0; i2 < numStates; i2++) {
                        for (int k = 0; k < sigmaSize ; k++) {
                            double num = 0;
                            double denom = 0;

                            for (int t = 0; t <= T - 1; t++) {
                                double g = gamma(i2, t, sampleData, fwd, bwd);
                                num += g * (k == sampleData[t] ? 1 : 0);
                                denom += g;
                            }
                            b1[i2][k] = divide(num, denom);
                        }
                    }
                    userPi[u] = pi1;
                    userTransProb[u] = a1;
                    userEmissionProb[u] = b1;
                }
            }
        }
    }


    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {

        double x = 0.0;

        double h = timeMatrix.get(userIdx, itemIdx) / 1000;
        int t = (int) Math.round(h / 6);

        //t for next period
        if((t + 1) == 3)
        {
           t = 0;
        }
        else
        {
           t++;
        }

        double[][] fwd;
        double[][] bwd;
        fwd = forwardProc(userSampleData[userIdx]);
        bwd = backwardProc(userSampleData[userIdx]);


        for (int j = 0; j < numStates; j++) {
            for (int k = 0; k < numStates; k++) {
              x+= p(t, j, k, userSampleData[userIdx], fwd, bwd);
            }
        }

        return x;
    }

    /**
     * calculation of Forward-Variables f(i,t) for state i at time
     * t for output sequence O with the current HMM parameters
     *
     * @param o the output sequence O
     * @return an array f(i,t) over states and times, containing
     * the Forward-variables.
     */
    public double[][] forwardProc(int[] o) {
        int T = o.length;
        double[][] fwd = new double[numStates][T];

    /* initialization (time 0) */
        for (int i = 0; i < numStates; i++)
            fwd[i][0] = pi[i] * emissionProb[i][o[0]];

    /* induction */
        for (int t = 0; t <= T - 2; t++) {
            for (int j = 0; j < numStates; j++) {
                fwd[j][t + 1] = 0;
                for (int i = 0; i < numStates; i++)
                    fwd[j][t + 1] += (fwd[i][t] * emissionProb[i][j]);
                fwd[j][t + 1] *= emissionProb[j][o[t + 1]];
            }
        }

        return fwd;
    }

    /**
     * calculation of  Backward-Variables b(i,t) for state i at time
     * t for output sequence O with the current HMM parameters
     *
     * @param o the output sequence O
     * @return an array b(i,t) over states and times, containing
     * the Backward-Variables.
     */
    public double[][] backwardProc(int[] o) {
        int T = o.length;
        double[][] bwd = new double[numStates][T];

    /* initialization (time 0) */
        for (int i = 0; i < numStates; i++)
            bwd[i][T - 1] = 1;

    /* induction */
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < numStates; i++) {
                bwd[i][t] = 0;
                for (int j = 0; j < numStates; j++) {
                    double val = (bwd[j][t + 1] * transProb[i][j] * emissionProb[j][o[t + 1]]);
                    bwd[i][t] += val;
                }
            }
        }

        return bwd;
    }

    /**
     * calculation of probability P(X_t = s_i, X_t+1 = s_j | O, m).
     *
     * @param t   time t
     * @param i   the number of state s_i
     * @param j   the number of state s_j
     * @param o   an output sequence o
     * @param fwd the Forward-Variables for o
     * @param bwd the Backward-Variables for o
     * @return P
     */
    public double p(int t, int i, int j, int[] o, double[][] fwd, double[][] bwd) {
        double num;
        if (t == o.length - 1)
            num = fwd[i][t] * transProb[i][j];
        else
            num = fwd[i][t] * transProb[i][j] * emissionProb[j][o[t + 1]] * bwd[j][t + 1];

        double denom = 0;

        for (int k = 0; k < numStates; k++)
            denom += (fwd[k][t] * bwd[k][t]);

        return divide(num, denom);
    }

    /**
     * computes gamma(i, t)
     */
    public double gamma(int i, int t, int[] o, double[][] fwd, double[][] bwd) {
        double num = fwd[i][t] * bwd[i][t];
        double denom = 0;

        for (int j = 0; j < numStates; j++)
            denom += fwd[j][t] * bwd[j][t];

        return divide(num, denom);
    }


    /**
     * divides two doubles. 0 / 0 = 0!
     */
    public double divide(double n, double d) {
        if (n == 0)
            return 0;
        else
            return n / d;
    }

}

class SequenceRecord {
    private int user;
    private int item;
    private List<Integer> states;
    private List<Double> transitionProbs;
    private List<Double> emissionProbs;

    SequenceRecord(int user, int item, int state) {
        this.user = user;
        this.item = item;
        //this.state = state;
    }

    public int getUser() {
        return user;
    }

    public void setUser(int user) {
        this.user = user;
    }

    public int getItem() {
        return item;
    }

    public void setItem(int item) {
        this.item = item;
    }

    public List<Integer> getStates() {
        return states;
    }

    public void setStates(List<Integer> states) {
        this.states = states;
    }

    public List<Double> getTransitionProbs() {
        return transitionProbs;
    }

    public void setTransitionProbs(List<Double> transitionProbs) {
        this.transitionProbs = transitionProbs;
    }

    public List<Double> getEmissionProbs() {
        return emissionProbs;
    }

    public void setEmissionProbs(List<Double> emissionProbs) {
        this.emissionProbs = emissionProbs;
    }
}

class UserItemStateCount {

    private int user;
    private int item;
    private int state;
    private int count;

    public UserItemStateCount(int user, int item, int state, int count) {
        this.user = user;
        this.item = item;
        this.state = state;
        this.count = count;
    }

    public int getUser() {
        return user;
    }

    public int getItem() {
        return item;
    }

    public int getState() {
        return state;
    }

    public int getCount() {
        return count;
    }

    public void iterateCount() {
        count++;
    }
}