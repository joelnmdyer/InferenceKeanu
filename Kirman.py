import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

np.random.seed(seed=18455)

T = 40
N = 10
y = np.zeros(T, dtype=np.int)
y[0] = int(N/2.)
n = y[0]
e = 0.15
d = 0.3
p = np.random.uniform(0., 1., size=T-1)

for i in range(1, T):
    down = n/N*(e + (1-d)*(N-n)/(N-1))
    up = (1-n/N)*(e + (1-d)*n/(N-1))
    assert 0 <= down + up < 1
    if p[i-1] < down:
        n = n - 1
    elif down <= p[i-1] < down + up:
        n = n + 1
    assert 0 <= n <= N
    y[i] = n

# Probabilistic model
with pm.Model() as model:
    # These are the two parameters in the model
    eP = pm.Uniform('e', 0., 1.)
    dP = pm.Uniform('d', 0., 1.)
    # Initial number of ants in state 1 could be anything between 0 and N, inclusive
    y0 = pm.DiscreteUniform('y0', 0, N, observed=y[0])
    for t in range(1, T):
        # Get previous state
        n = model.named_vars['y{0}'.format(t-1)]
        downP = n / N * (eP + (1 - dP) * (N - n) / (N - 1))
        upP = (1 - n / N) * (eP + (1 - dP) * n / (N - 1))
        ps = tt.stack([downP, 1-downP-upP, upP])
        # Check that the real change in the data is captured by the support of the Categorical variable that we will
        # construct below
        assert y[t]-y[t-1]+1 in [0, 1, 2]
        # Observing the actual state of the model would mean observing Z = n + y_t - 1, where y_t is defined as below.
        # However, there are two problems with this: 1) there doesn't seem to be a way to shift the support of a
        # Categorical variable to {n - 1, n, n + 1}, as would be the most natural way to model Z; 2) even if there was
        # a more obvious way to shift the support/create Z some other way, we're creating a latent variable y_t for each
        # Z created in each time step, which slows things down. So I thought I would just observe the initial state and
        # observe the change at each time step, which is equivalent but which eliminates unnecessary latents.
        y_t = pm.Categorical('y{0}'.format(t), p=ps, observed=y[t]-y[t-1]+1)
    print(model.free_RVs)
    model.check_test_point()
    trace = pm.sample(1000, cores=4, tune=1000)
    pm.summary(trace)
    fig, axs = plt.subplots(2, 2)
    pm.traceplot(trace, varnames=['e', 'd'], ax=axs)
