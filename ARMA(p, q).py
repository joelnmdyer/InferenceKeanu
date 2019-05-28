import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

np.random.seed(seed=1848)

T = 10
y = np.zeros(T)
y[1] = 1.
e = np.random.normal(0., 1., size=T)

# Generate data
for i in range(2, T):
    y[i] = 0.7 * y[i-1] + 0.1 * y[i-2] + 0.2 * e[i-1] + 0.2 * e[i-2] + e[i]

#plt.plot(y);
#plt.show();

tau = 1.0
sig = 1.0
# Build probabilistic model
with pm.Model() as model:
    # These are the four AR and MA coefficients for the ARMA(2,2) model
    a1 = pm.Uniform('a1', 0., 1.)
    a2 = pm.Uniform('a2', 0., 1.)
    b1 = pm.Uniform('b1', 0., 1.)
    b2 = pm.Uniform('b2', 0., 1.)
    y0 = pm.Normal('y0', mu=0., sd=1., observed=y[0])
    y1 = pm.Normal('y1', mu=1., sd=1., observed=y[1])
    e0 = pm.Normal('e0', mu=0., sd=sig)
    e1 = pm.Normal('e1', mu=0., sd=sig)
    for t in range(2, T):
        eps = pm.Normal('e{0}'.format(t), mu=0., sd=sig)
        last = model.named_vars['y{0}'.format(t-1)]
        lastlast = model.named_vars['y{0}'.format(t-2)]
        laste = model.named_vars['e{0}'.format(t-1)]
        lastlaste = model.named_vars['e{0}'.format(t-2)]
        y_m = pm.Normal('y{0}'.format(t), mu=a1 * last + a2 * lastlast + b1 * laste + b2 * lastlaste + e, sd=1.,
                        observed=y[t])
    print(model.free_RVs)
    trace = pm.sample(4000, cores=4, tune=1000)
    pm.summary(trace)
    fig, axs = plt.subplots(4, 2)
    pm.traceplot(trace, varnames=['a1', 'a2', 'b1', 'b2'], ax=axs)
    plt.show()
