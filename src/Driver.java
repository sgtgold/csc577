import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import net.librec.math.algorithm.Randoms;

import java.io.FileInputStream;
import java.util.Properties;

/**
 * Created by rburke on 12/5/2016.
 */


public class Driver {

    // Change this to load a different configuration file.
    //public static String CONFIG_FILE = "conf/mac-time-svd.properties";
    public static String CONFIG_FILE = "conf/time-svd.properties";

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        String confFilePath = CONFIG_FILE;

        conf.set("rec.recommender.verbose","true");
        Properties prop = new Properties();
        prop.load(new FileInputStream(confFilePath));
        for (String name : prop.stringPropertyNames()) {
            conf.set(name, prop.getProperty(name));
        }

        Randoms.seed(20160129);
        RecommenderJob job = new RecommenderJob(conf);
        job.runJob();
        System.out.print("Finished");
    }
}
