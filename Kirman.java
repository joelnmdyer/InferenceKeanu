import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

public class Kirman {

    public static void main(String[] args) {

        // Number of agents
        int N = 100;
        // Number in state 1
        double n = 50;
        // Model parameters
        double eps = 0.15;
        double del = 0.3;
        // Number of times steps to simulate
        int T = 1000;
        // Generate data
        boolean[][] data = kirmanModel(eps, del, N, n, T);
        // Construct the probabilistic model:
        // Priors for parameters
        UniformVertex eP = new UniformVertex(0., 1.);
        UniformVertex dP = new UniformVertex(0., 0.5);
        // This is a 2D array, where rows are labelled by time step value and
        // columns 0, 1, 2 are boolean values indicating the change in the
        // number of agents in state 1 at that time step. Element 0 being true
        // indicates that n -> n - 1 that time step; element 1 being true mean
        // n -> n that time step (i.e. no change); 2 true means n -> n + 1 (i.e.
        // an agent converts from state 2 to state 1)
        List<List<BooleanVertex>> kirmanProbModel = buildModel(eP, dP, N, T);
        // Observe data
        for (int j = 0; j < T; j++) {
            List<BooleanVertex> Yj = kirmanProbModel.get(j);
            for (int l = 0; l < 3; l++) {
                Yj.get(l).observe(data[j][l]);
            }
        }
        // I think this updates the graph given that kirmanProbModel vertex values
        // have been set. Do I need to do this?
        //VertexValuePropagation.cascadeUpdate(kirmanProbModel);
        // Estimate parameters
        // "The Optimizer mutates the values of the graph while finding the
        // most probable values and leaves the graph in its most optimal state.
        // Therefore, to find the most probable value of a vertex once, simply
        // get the value of the vertex."
        BayesianNetwork net = new BayesianNetwork(eP.getConnectedGraph());
        // Keanu decides whether or not to use gradient or non-gradient
        // optimiser
        Optimizer graphOptimizer = Keanu.Optimizer.of(net);
        graphOptimizer.maxAPosteriori();

        double calculatedeP = eP.getValue().scalar();
        double calculateddP = dP.getValue().scalar();
        System.out.println("e = " + calculatedeP);
        System.out.println("d = " + calculateddP);

    }

    private static boolean[][] kirmanModel(double e, double d, int N, double n,
                                     int T) {

        // data will just be a sequence of changes (-1, 0, 1) rather than the
        // actual values of n
        boolean[][] data = new boolean[T][3];
        // No change in the first step: initial value is set
        data[0][0] = false;
        data[0][1] = true;
        data[0][2] = false;
        for (int t = 1; t < T; t++) {
            // Get the change in n that happened last time step
            int step = sum(data[t-1]);
            // Get the transition probabilities given the current state
            double[] probs = getProbs(e, d, N, n+step);
            // Determine change in n this time step
            UniformVertex choice = new UniformVertex(0, 1);
            double val = choice.sample().getValue(0);
            data[t][0] = false;
            data[t][1] = false;
            data[t][2] = false;
            if (val < probs[0]) {
                System.out.println("Down");
                data[t][0] = true;
            }
            else if (val < probs[0] + probs[1]) {
                System.out.println("Stay");
                data[t][1] = true;
            }
            else {
                System.out.println("Up");
                data[t][2] = true;
            }
            System.out.println("t = " + t + ": " + data[t]);
        }

        return data;

    }

    private static int sum(boolean[] timeSlice) {

        // This gives the direction of the step taken in a time slice of the
        // nested step list: val will be -1 if n -> n - 1, +1 if n -> n + 1, 0
        // otherwise
        int val = 0;
        val += timeSlice[0] ? -1 : 0;
        val += timeSlice[2] ?  1 : 0;
        return val;

    }

    private static double[] getProbs(double e, double d, int N, double n) {

        // Get the transition probabilities
        double[] probs = new double[3];
        // Probability of moving down
        probs[0] = n*(e + (1-d)*(N-n)/(N-1))/N;
        // Probability of moving up
        probs[2] = (1-n/N)*(e + (1-d)*n/(N-1));
        // Probability of no change
        probs[1] = 1 - (probs[0] + probs[2]);
        System.out.println("n -> n - 1: " + probs[0] +
                "; n -> n: " + probs[1] +
                "; n -> n + 1: " + probs[2]);
        // Don't know if this is the correct syntax but have observed no
        // violations yet just from the print statements
        assert 1 >= probs[0] & probs[0] >= 0;
        assert 1 >= probs[1] & probs[1] >= 0;
        assert 1 >= probs[2] & probs[2] >= 0;
        assert probs[0] + probs[1] + probs[2] <= 1;

        return probs;

    }

    private static List<List<BooleanVertex>> buildModel(DoubleVertex e,
                                                        DoubleVertex d,
                                                        int N, int T) {

        // Construct probabilistic model for Kirman ABM
        // graphTimeSteps is a T x 3 nested list of boolean vertices. Elements
        // 0, 1, 2 indicate changes n -> n - 1, n -> n, n -> n + 1
        List<List<BooleanVertex>> graphTimeSteps = new ArrayList<>();
        // Uniform distribution over initial state
        DoubleVertex n0 = new UniformVertex(0, N);
        DoubleVertex n = n0;
        for (int t = 0; t < T; t++) {
            // Get transition probabilities in a probabilistic sense
            DoubleVertex[] probs = getProbProbs(e, d, n, N);
            // Decide which direction to move in. Not sure how best to capture
            // dependencies between down, stay, and up below other than having
            // a single probabilistic vertex p whose value determines the
            // probability that down, stay, up should evaluate to true
            DoubleVertex p = new UniformVertex(0., 1.);
            BooleanVertex down = p.lessThan(probs[0]);
            BooleanVertex up   = p.greaterThan(probs[0]).and(p.lessThan(probs[0].plus(probs[2])));
            BooleanVertex stay = down.not().and(up.not());
            graphTimeSteps.add(Arrays.asList(down, stay, up));
            // Determine the change in state n
            DoubleVertex move = If.isTrue(down)
                    .then(
                            -1.0
                    ).orElse(
                            If.isTrue(up)
                                    .then(1.0)
                                    .orElse(0.0)
                    );
            // Update state probabilistically
            n = n.plus(move);
        }

        return graphTimeSteps;

    }

    private static DoubleVertex[] getProbProbs(DoubleVertex e, DoubleVertex d, DoubleVertex n, int N) {

        // Transition probabilities. Probabilistic mirror of getProbs above
        DoubleVertex[] probs = new DoubleVertex[3];
        probs[0] = n.times(e.plus(d.minus(1).times(n.minus(N)).div(N-1))).div(N);
        probs[2] = n.div(N).minus(1).times(-1).times(e.plus(d.minus(1).times(n).div(N-1)));
        probs[1] = probs[0].plus(probs[2]).minus(1).times(-1);

        return probs;

    }

    // Another idea I had was to create a CategoricalVertex which generated
    // -1, 0, +1 with probabilities determined by the transition probabilities
    // generated by getProbProbs below. I would have then just updated the
    // state by doing n.plus(the instance of CategoricalVertex generated at
    // time step t), and then just observe the state n at each time step rather
    // than the three boolean values indicating the change in n at each time
    // step. However, keanu isn't happy with me doing arithmetic with n (a
    // DoubleVertex) and a CategoricalVertex, and I'm not sure how to get
    // around that.
/*    public static CategoricalVertex<Double, GenericTensor<Double>> jump(DoubleVertex e, DoubleVertex d, DoubleVertex n, int N) {

        LinkedHashMap<Double, DoubleVertex> frequency = new LinkedHashMap<>();
        DoubleVertex[] probs = getProbProbs(e, d, n, N);
        frequency.put(-1., probs[0]);
        frequency.put(0., probs[1]);
        frequency.put(1., probs[2]);

        return new CategoricalVertex<>(frequency);
    }*/

}