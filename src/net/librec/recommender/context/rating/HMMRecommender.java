package net.librec.recommender.context.rating;

import net.librec.common.LibrecException;
import java.util.*;

import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.AbstractRecommender;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jake on 3/4/2017.
 */
public class HMMRecommender extends AbstractRecommender {

    /** number of states */
    private int numStates;

    /** size of output vocabulary */
    private int sigmaSize;

    /** size of output vocabulary */
    private int numSteps;

    /** initial state probabilities */
    private double pi[];

    /** transition probabilities */
    private double transProb[][];

    /** emission probabilities */
    private double emissionProb[][];

    /** User Artist Id lookups**/
    private List<SequenceRecord> userStateLookup;

    /** User Artist Id lookups**/
    private HashMap<Integer,List<Integer>> userStateData;

    /**
     * matrix of time stamp
     */
    private static SparseMatrix timeMatrix;


    private  List<UserItemStateCount> userItemStateCounts ;


    @Override
    protected void setup() throws LibrecException {
        super.setup();
        numStates = 4;//conf.getInt("rec.hmm.numStates", 6);
        sigmaSize = 10;
        ;//conf.getInt("rec.hmm.sigmaSize", 12);
        numSteps = 100;//conf.getInt("rec.hmm.numSteps", 100);
        pi = new double[numStates];
        transProb = new double[numStates][numStates];
        emissionProb = new double[numStates][sigmaSize];

        timeMatrix = (SparseMatrix) getDataModel().getDatetimeDataSet();
        userStateLookup = new ArrayList<>();
        userStateData = new HashMap<Integer,List<Integer>>();
        userItemStateCounts = new ArrayList<UserItemStateCount>();



    }
    @Override
    protected void trainModel() throws LibrecException {

        List<Integer> stateData = new ArrayList<>();
        List<Integer> states = new ArrayList<Integer>();
        for (int u = 0; u < numUsers; u++) {
            List<Integer> userItems = trainMatrix.getColumns(u);
            for (int id = 0; id < userItems.size(); id++) {
                if (id < numStates) {
                    userStateLookup.add(new SequenceRecord(u, userItems.get(id), id));
                }
            }
        }

        //Merge train data
        for (int u = 0; u < numUsers; u++) {
            for (int i = 0; i < numItems; i++) {
                for (int t = 0; t < timeMatrix.getRows(u).size(); t++) {
                    final int i1 = i;
                    final int u1 = u;


                    if (userStateLookup.stream().filter(x -> x.getItem() == i1 && x.getUser() == u1).findFirst().isPresent()) {
                        SequenceRecord record = userStateLookup.stream().filter(x -> x.getItem() == i1 && x.getUser() == u1).findFirst().get();

                        double h = timeMatrix.get(u, i) / 1000;

                        final int state = (int) Math.round(h / 6);
                        if (userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getItem() == i1 && x.getState() == state).findFirst().isPresent()) {
                            UserItemStateCount uisc = userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getItem() == i1 && x.getState() == state).findFirst().get();
                            uisc.iterateCount();
                        } else {
                            UserItemStateCount uisc = new UserItemStateCount(u1, i1, state, 1);
                            userItemStateCounts.add(uisc);
                        }
                    }
                }
            }
        }

        //Can move this up once written
        //Loop through users
        for (int u = 0; u < numUsers; u++) {
            //Loop through items
            //User Level aggregates
            double [][] stateCounts = new double[numUsers][numStates];
            double [] favArtistForState = new double[numStates];
            for (int i = 0; i < numItems; i++) {
                //Item level aggregates

                //double userItemStateTotal = 0.0;
                //Loop through states
                for (int s = 0; s < numStates; s++) {
                    final int u1 = u;
                    final int i1 = i;
                    final int s1 = s;
                    //fave artist
                    if (userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getItem() == i1 && x.getState() == s1).findFirst().isPresent()) {
                        Iterator<UserItemStateCount> userStateItr = userItemStateCounts.stream().filter(x -> x.getUser() == u1 && x.getState() == s1).iterator();
                        while (userStateItr.hasNext()) {
                            UserItemStateCount uisc = userStateItr.next();
                            if (favArtistForState[s] < uisc.getCount()) {
                                favArtistForState[s] = uisc.getCount();
                            }
                            stateCounts[u][s] = uisc.getCount();
                        }
                    }
                }
            }
            //Initial State Probabilities DO NOT DELETE
          /*  if(stateCounts[u].length > 0) {
                double total = 0.0;
                for (int s1 = 0; s1 < stateCounts[u].length  ; s1++) {
                    total += stateCounts[u][s1];
                }
                if (total > 0.0) {
                    for (int s2 = 0; s2 < stateCounts[u].length  ; s2++) {
                        double initialState;
                        initialState = stateCounts[u][s2] / total;
                        System.out.println("User: "+ u +" State: " + s2 + " Percent: " + initialState);
                    }
                }
            }*/
            //End Initial State
            //forward state changes
            double[][] stateChangeMatrix = new double[numStates][numStates-1];
            double totalNumberOfStateChanges = 0;
            for(int s3 = 0; s3 < numStates - 1; s3++)
            {
                for(int s4 = 0; s4 < numStates - 1; s4++)
                {
                    if(favArtistForState[s3] != favArtistForState[s4])
                    {
                        stateChangeMatrix[s3][s4] = 1.0;
                        totalNumberOfStateChanges++;
                    }
                }


            }
            //backwards state changes

            for(int s7 = numStates - 1; s7 > 0; s7--)
            {
                for(int s8 = numStates - 1; s8 > 0; s8--)
                {
                    if(favArtistForState[s7] != favArtistForState[s8])
                    {
                        stateChangeMatrix[s7][s8] = 1.0;
                        totalNumberOfStateChanges++;
                    }
                }
            }


            for(int s5 = 0; s5 < numStates - 1; s5++)
            {
                for(int s6 = 0; s6 < numStates - 1; s6++)
                {
                    if(totalNumberOfStateChanges != 0)
                    {
                        System.out.println("User: "+ u +" State_1: " + s5 +" State_2: " + s6 + " Probability: " + (stateChangeMatrix[s5][s6] / totalNumberOfStateChanges) );
                    }
                    else {
                        System.out.println("User: " + u + " State_1: " + s5 + " State_2: " + s6 + " Probability: " + 0.0);
                    }
                }
            }
            for(int s9 = numStates - 1; s9 > 0; s9--)
            {
                for(int s10 = numStates - 1; s10 > 0; s10--)
                {
                    if(favArtistForState[s9] != favArtistForState[s10])
                    {
                        stateChangeMatrix[s9][s10] = 1.0;
                        if(totalNumberOfStateChanges != 0)
                        {
                            System.out.println("User: "+ u +" State_1: " + s9 +" State_2: " + s10 + " Probability: " + (stateChangeMatrix[s9][s10] / totalNumberOfStateChanges) );
                        }
                        else {
                            System.out.println("User: " + u + " State_1: " + s9 + " State_2: " + s10 + " Probability: " + 0.0);
                        }
                    }
                }
            }
        }
        System.out.println("Finished");


                    //sorting for sequence
                    //insert at beginning
                        /*if (currseq <= pastseq) {
                            states.add(0, record.getState());
                        }
                        //insert at end
                        if (currseq > pastseq) {
                            states.add(states.size() - 1, record.getState());
                        }
                        pastseq = currseq;
                        */


    }
        /*
         for(int u = 0; u < numUsers; u++) {
             int T = userStateData.get(u).size();
             double[][] fwd;
             double[][] bwd;

             double pi1[] = new double[numStates];
             double a1[][] = new double[numStates][numStates];
             double b1[][] = new double[numStates][sigmaSize];

             for (int s = 0; s < numSteps; s++) {
      // calculation of Forward- und Backward Variables from thecurrent model
                List<Integer> o = userStateData.get(u);
                 fwd = forwardProc(o);
                 bwd = backwardProc(o);

      // re-estimation of initial state probabilities
                 for (int i = 0; i < numStates; i++)
                    pi1[i] = gamma(i, 0, userStateData.get(u), fwd, bwd);

      // re-estimation of transition probabilities
                 for (int i = 0; i < numStates; i++) {
                     for (int j = 0; j < numStates; j++) {
                         double num = 0;
                         double denom = 0;
                         for (int t = 0; t <= T - 1; t++) {
                             num += p(t, i, j, userStateData.get(u), fwd, bwd);
                             denom += gamma(i, t, userStateData.get(u), fwd, bwd);
                         }
                         a1[i][j] = divide(num, denom);
                     }
                 }

      //re-estimation of emission probabilities
                 for (int i = 0; i < numStates; i++) {
                     for (int k = 0; k < sigmaSize; k++) {
                         double num = 0;
                         double denom = 0;

                         for (int t = 0; t <= T - 1; t++) {
                             double g = gamma(i, t, userStateData.get(u), fwd, bwd);
                             num += g * (k == userStateData.get(u).get(t) ? 1 : 0);
                             denom += g;
                         }
                         b1[i][k] = divide(num, denom);
                     }
                 }
                 pi = pi1;
                 transProb = a1;
                 emissionProb = b1;
             }

         }*/
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {

        for (int i = 0; i < numStates; i++)
            System.out.println("pi(" + i + ") = " + pi[i]);
        System.out.println();

        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numStates; j++)
                System.out.print("transProb(" + i + "," + j + ") = " +
                        transProb[i][j] + "  ");
            System.out.println();
        }

        System.out.println();
        for (int i = 0; i < numStates; i++) {
            for (int k = 0; k < sigmaSize; k++)
                System.out.print("emissionProb(" + i + "," + k + ") = " +
                        emissionProb[i][k] + "  ");
            System.out.println();
        }



        return 1.0;
    }
    /** calculation of Forward-Variables f(i,t) for state i at time
     t for output sequence O with the current HMM parameters
     @param o the output sequence O
     @return an array f(i,t) over states and times, containing
     the Forward-variables.
     */
    public double[][] forwardProc(List<Integer> o) {
        int T = o.size();
        double[][] fwd = new double[numStates][T];

    /* initialization (time 0) */
        for (int i = 0; i < numStates; i++)
            fwd[i][0] = pi[i] * emissionProb[i][o.get(0)];

    /* induction */
        for (int t = 0; t <= T-2; t++) {
            for (int j = 0; j < numStates; j++) {
                fwd[j][t+1] = 0;
                for (int i = 0; i < numStates; i++)
                    fwd[j][t+1] += (fwd[i][t] * transProb[i][j]);
                fwd[j][t+1] *= emissionProb[j][o.get(t+1)];
            }
        }

        return fwd;
    }

    /** calculation of  Backward-Variables emissionProb(i,t) for state i at time
     t for output sequence O with the current HMM parameters
     @param o the output sequence O
     @return an array emissionProb(i,t) over states and times, containing
     the Backward-Variables.
     */
    public double[][] backwardProc(List<Integer> o) {
        int T = o.size();
        double[][] bwd = new double[numStates][T];


    /* initialization (time 0) */
        for (int i = 0; i < numStates; i++)
            bwd[i][T-1] = 1;

    /* induction */
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < numStates; i++) {
                bwd[i][t] = 0.0;
                double x = 0.0;
                for (int j = 0; j < numStates; j++) {
                    double ep = emissionProb[j][o.get(t+1)];
                    double tp = transProb[i][j];
                    double nb = bwd[j][t + 1];

                     x += (nb * tp * ep);

                    bwd[i][t] = x;
                }
            }
        }

        return bwd;
    }

    /** calculation of probability P(X_t = s_i, X_t+1 = s_j | O, m).
     @param t time t
     @param i the number of state s_i
     @param j the number of state s_j
     @param o an output sequence o
     @param fwd the Forward-Variables for o
     @param bwd the Backward-Variables for o
     @return P
     */
    public double p(int t, int i, int j, List<Integer> o, double[][] fwd, double[][] bwd) {
        double num;
        if (t == o.size() - 1)
            num = fwd[i][t] * transProb[i][j];
        else
            num = fwd[i][t] * transProb[i][j] * emissionProb[j][o.get(t+1)] * bwd[j][t+1];

        double denom = 0;

        for (int k = 0; k < numStates; k++)
            denom += (fwd[k][t] * bwd[k][t]);

        return divide(num, denom);
    }

    /** computes gamma(i, t) */
    public double gamma(int i, int t, List<Integer> o, double[][] fwd, double[][] bwd) {
        double num = fwd[i][t] * bwd[i][t];
        double denom = 0;

        for (int j = 0; j < numStates; j++)
            denom += fwd[j][t] * bwd[j][t];

        return divide(num, denom);
    }

    /** divides two doubles. 0 / 0 = 0! */
    public double divide(double n, double d) {
        if (n == 0)
            return 0;
        else
            return n / d;
    }

    public class SequenceRecord {
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
    public class UserItemStateCount {

        private int user;
        private int  item;
        private int  state;
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
        private void iterateCount() {
            count++;
        }
    }
}