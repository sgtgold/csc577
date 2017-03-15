package net.librec.recommender.context.rating;

import net.librec.common.LibrecException;

import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.AbstractRecommender;

import java.util.ArrayList;
import java.util.List;

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
     * initial timeFrame probabilities
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
     * user initial timeFrame probabilities cache
     */
    private double userPi[][];
    /**
     * user timeFrame data cache
     */
    private int userStateData[][];

    /**
     * matrix of time stamp
     */
    private static SparseMatrix timeMatrix;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        numStates = conf.getInt("rec.hmm.numStates", 4);
        sigmaSize = conf.getInt("rec.hmm.sigmaSize", 10);
        numSteps = conf.getInt("rec.hmm.numSteps", 100);
        pi = new double[numStates];
        transProb = new double[numStates][numStates];
        emissionProb = new double[numStates][sigmaSize];


        userPi = new double[numUsers][numStates];
        userTransProb = new double[numUsers][numStates][numStates];
        userEmissionProb = new double[numUsers][numStates][sigmaSize];
        userStateData = new int[numUsers][numStates];

        for (int s = 0; s < numStates; s++) {
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
    }

    @Override
    protected void trainModel() throws LibrecException {
        List<UserItemTimeFrameCount> userItemStateCounts = new ArrayList<>();
        //Define the time frames and find the user's most popular argument
        for (int u = 0; u < numUsers; u++) {
            List<Integer> oi = new ArrayList<>();
            for (int i = 0; i < sigmaSize; i++) {
                int max = 0;
                int favArtistForTimeFrame = -1;
                for (int t = 0; t < timeMatrix.getRows(u).size(); t++) {
                    final int i1 = i;
                    final int u1 = u;
                    double h = timeMatrix.get(u, i) / 1000;
                    final int timeFrame = (int) Math.round(h / 6);
                    if (userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getItem() == i1 && x.getTimeFrame() == timeFrame).findFirst().isPresent()) {
                        UserItemTimeFrameCount uisc = userItemStateCounts.stream().filter(x -> x.getUser() == u1
                                && x.getItem() == i1
                                && x.getTimeFrame() == timeFrame).findFirst().get();
                        uisc.iterateCount();
                        if (uisc.getCount() > max) {
                            max = uisc.getCount();
                            favArtistForTimeFrame = uisc.getItem();
                        }
                    } else {
                        UserItemTimeFrameCount uisc = new UserItemTimeFrameCount(u1, i1, timeFrame, 1);
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
                userStateData[u] = sampleData;
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

                    // re-estimation of initial timeFrame probabilities
                    for (int k = 0; k < numStates; k++) {
                        pi1[k] = gamma(k, 0, sampleData, fwd, bwd);
                    }
                    // re-estimation of transition probabilities
                    for (int i1 = 0; i1 < numStates; i1++) {
                        for (int j = 0; j < numStates; j++) {
                            double num = 0;
                            double denom = 0;
                            for (int t = 0; t <= T - 1; t++) {
                                num += probability(t, i1, j, sampleData, fwd, bwd);
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
        fwd = forwardProc(userStateData[userIdx]);
        bwd = backwardProc(userStateData[userIdx]);


        for (int j = 0; j < numStates; j++) {
            for (int k = 0; k < numStates; k++) {
              x+= probability(t, j, k, userStateData[userIdx], fwd, bwd);
            }
        }

        return x;
    }

    /**
     * calculation of Forward-Variables f(i,t) for timeFrame i at time
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
     * calculation of  Backward-Variables b(i,t) for timeFrame i at time
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
     * @param i   the number of timeFrame s_i
     * @param j   the number of timeFrame s_j
     * @param o   an output sequence o
     * @param fwd the Forward-Variables for o
     * @param bwd the Backward-Variables for o
     * @return P
     */
    public double probability(int t, int i, int j, int[] o, double[][] fwd, double[][] bwd) {
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

class UserItemTimeFrameCount {

    private int user;
    private int item;
    private int timeFrame;
    private int count;

    public UserItemTimeFrameCount(int user, int item, int timeFrame, int count) {
        this.user = user;
        this.item = item;
        this.timeFrame = timeFrame;
        this.count = count;
    }

    public int getUser() {
        return user;
    }

    public int getItem() {
        return item;
    }

    public int getTimeFrame() {
        return timeFrame;
    }

    public int getCount() {
        return count;
    }

    public void iterateCount() {
        count++;
    }
}