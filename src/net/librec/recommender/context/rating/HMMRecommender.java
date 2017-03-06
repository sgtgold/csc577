package net.librec.recommender.context.rating;

import net.librec.common.LibrecException;
import net.librec.recommender.AbstractRecommender;

/**
 * Created by jake on 3/4/2017.
 */
public class HMMRecommender extends AbstractRecommender {

    @Override
    protected void setup() throws LibrecException {

    }
    @Override
    protected void trainModel() throws LibrecException {

    }
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        return 1.0;
    }
}
