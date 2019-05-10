import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.templating.SequenceItem;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LogNormalVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class aRModel {

    public static void main(String[] args) {

        // Number of time steps to simulate
        int T = 1000;
        // Coefficient values
        double[] pars = new double[2];
        pars[0] = 0.5;
        pars[1] = 0.1;
        // Take zero offset
        double c = 0;
        // Noise at each step is distributed as standard normal
        GaussianVertex eps = new GaussianVertex(0., 1.);
        // Simulated data
        double[] data = new double[T];
        // Initial conditions
        double x0 = 0.;
        double x1 = 1.;
        data[0] = x0;
        data[1] = x1;

        // Simulate data
        for (int t = 2; t < T; t++) {
            DoubleTensor e = eps.sampleWithShape(new long[]{1, 1});
            // Need index in getValue call otherwise it gives an error saying
            // indices must have same dimensions as array rank or something
            data[t] = e.getValue(0) + pars[0]*data[t-1] + pars[1]*data[t-2] + c;
            //System.out.println(data[t]);
        }
        // **** Above works for generating the AR(2) time series data **** //
        // Actually I haven't checked that this actually has all the correct
        // characteristics but it produces numbers that look ok so

        // Construct probabilistic model with keanu
        // Recall AR(2) process is y_t = a*y_{t-1} + b*y_{t-2} + c(=0 here) + eps
        UniformVertex a = new UniformVertex(0., 1.);
        UniformVertex b = new UniformVertex(0., 1.);
        // Standard deviation for normally-distributed noise. But actually you
        // can't just make one vertex for the entire graph, since they are
        // independent/distinct random variables even if they have the same
        // distribution
        UniformVertex s = new UniformVertex(0., 2.);
        // Assume stationary distribution
        // AR(2) process has Y_t = \sum_{i=0}^{\infty} \psi_i \epsilon_{t-i}
        // (see Brockwell and Davis (1991) Time Series: Theory and Methods
        // §3.3 and Wold representation) with \psi_0 = 1. So Y_0 \sim
        // \epsilon_0 and Y_1 \sim \epsilon_1 + \psi_1 \epsilon_0
        GaussianVertex y0 = new GaussianVertex(0., s);
        // \psi_1 = a for AR(2)
        GaussianVertex y1 = new GaussianVertex(0., s.times(a.plus(1)));

        // Build the dependency structure (I hope/think this is what this does)
        List<GaussianVertex> ar2Process = new ArrayList<>();
        ar2Process.add(y0);
        ar2Process.add(y1);
        for (int j = 2; j < T; j++) {
            GaussianVertex d = new GaussianVertex(ar2Process.get(j-1)
                    .multiply(a)
                    .plus(ar2Process.get(j-2).multiply(b)), s);
            ar2Process.add(d);
        }

        // Observe your data
        for (int j = 0; j < T; j++) {
            DoubleVertex Yj = ar2Process.get(j);
            // This type of vertex does not support being observed? I guess
            // because, even though it's a function of only probabilistic
            // variables, it's deterministic given their values. So it's not a
            // probabilistic vertex for this reason. Could get around this by
            // just making it a GaussianVertex centered on its deterministic
            // value, just as a workaround?
            Yj.observe(data[j]);
        }

        // I think this updates the graph given that ar2Process vertex values
        // have been set
        VertexValuePropagation.cascadeUpdate(ar2Process);

        // The Optimizer mutates the values of the graph while finding the
        // most probable values and leaves the graph in its most optimal state.
        // Therefore, to find the most probable value of a vertex once, simply
        // get the value of the vertex.
        BayesianNetwork net = new BayesianNetwork(s.getConnectedGraph());
        // Keanu decides whether or not to use gradient or non-gradient
        // optimiser
        Optimizer graphOptimizer = Keanu.Optimizer.of(net);
        graphOptimizer.maxAPosteriori();

        double calculatedSDev = s.getValue().scalar();
        double calculatedA = a.getValue().scalar();
        double calculatedB = b.getValue().scalar();
        System.out.println("s = " + calculatedSDev);
        System.out.println("a = " + calculatedA);
        System.out.println("b = " + calculatedB);

    }

}





// The following was something I was working on but gave up to do my original
// simpler idea

        /*// Define the labels of vertices we will use in our Sequence. 3 vertices
        // since AR(2) requires two lags and noise term
        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");

        // Define a factory method that creates proxy vertices using the proxy
        // vertex labels and then uses these to define the computation graph
        // of the Sequence.
        // Note we have labeled the output vertices of this SequenceItem
        Consumer<SequenceItem> factory = sequenceItem -> {
            // Define the Proxy Vertices which stand in for a Vertex from the
            // previous SequenceItem. They will be automatically wired up when
            // you construct the Sequence i.e. these are the 'inputs' to our
            // SequenceItem
            DoubleProxyVertex x1Input = sequenceItem.addDoubleProxyFor(x1Label);
            DoubleProxyVertex x2Input = sequenceItem.addDoubleProxyFor(x2Label);

            // Full model is linear combination of previous two lags + noise
            // So this step is adding the noise term
            DoubleVertex noise = new GaussianVertex(0., s);
            DoubleVertex xOut = x1Input.multiply(a).plus(x2Input.multiply(b)).plus(noise).setLabel(x1Label);
            // Add 0 so compiler doesn't complain about changing label on proxy
            // vertices
            DoubleVertex xLag2 = x1Input.plus(0);
            xLag2.setLabel(x2Label);
            sequenceItem.addAll(xOut, xLag2);
        };

        // Create the starting values of our sequence. We observe 0. and 1.
        // for first two items
        DoubleVertex x1Start = new ConstantDoubleVertex(0.).setLabel(x1Label);
        DoubleVertex x2Start = new ConstantDoubleVertex(1.).setLabel(x2Label);
        VertexDictionary dictionary = SimpleVertexDictionary.of(x1Start, x2Start);

        Sequence sequence = new SequenceBuilder<Integer>()
                .withInitialState(dictionary)
                .named("AR(2)")
                .count(1)
                .withFactory(factory)
                .build();

        // We can now put all the vertices in the sequence into a Bayes Net:
        BayesianNetwork bayesNet = sequence.toBayesianNetwork();
        List<Vertex> allVs = bayesNet.getAllVertices();
        System.out.println(allVs);*/

//Optimizer optimizer = Keanu.Optimizer.of(bayesNet);
//optimizer.maxAPosteriori();