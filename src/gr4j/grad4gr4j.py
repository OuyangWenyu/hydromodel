import os
import time
from functools import partial
import pandas as pd
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad
from jax.experimental import optimizers
from jax.ops import index, index_update
from jax.nn.initializers import glorot_normal, normal
from sklearn import preprocessing as pp
import definitions
from src.gr4j.jax4gr4j import hydrograms, calculate_precip_store, calculate_evap_store, calculate_perc


def state_init():
    """initialize the state variables, and make normalization"""

    def init():
        # S0 = 0.6 * 320.11
        # R0 = 0.7 * 69.63
        # Pr0 = jnp.zeros(9)
        S0 = 0.06
        R0 = 0.007
        Pr0 = jnp.zeros(9)
        return [S0, Pr0, R0]

    return init


def param_init():
    """initialize the parameters, and make normalization"""

    def init():
        # x1 = 320.11
        # x2 = 2.42
        # x3 = 69.63
        # x4 = 1.39
        # x4_limit = 5
        # manually set it
        x1 = 0.1
        x2 = 0.001
        x3 = 0.01
        x4 = 1.39
        x4_limit = 5
        # We can use fixed unit hydrograph at first
        UH1, UH2 = hydrograms(x4_limit, x4)
        # UH1 and UH2 intrinsically belong to [0, 1], so we don't need to normalize it again
        return [x1, x2, x3, UH1, UH2]

    return init


def GR4J(p_init=param_init(), s_init=state_init()):
    def init_fun(rng, input_shape):
        """ Initialize the layers"""
        hidden = s_init()

        param = p_init()

        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], 1)
        return (output_shape,
                (hidden,
                 param,
                 ),
                )

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def apply_fun_scan(param, hidden, inp):
            """ Perform single step update of the network """
            # params
            x1 = param[0]
            x2 = param[1]
            x3 = param[2]
            UH1 = param[3]
            UH2 = param[4]
            # state_variables
            S = hidden[0]
            runoff_history = hidden[1]
            R = hidden[2]
            # forcings
            P = inp[0]
            E = inp[1]
            # Calculate net precipitation and evapotranspiration
            precip_difference = P - E
            precip_net = jnp.maximum(precip_difference, 0)
            evap_net = jnp.maximum(-precip_difference, 0)

            # x1 must be positive
            x1 = jnp.clip(x1, a_min=0.0001)
            # S cannot be larger than x1, either smaller than 0
            S = jnp.clip(S, a_min=0, a_max=x1)

            # Calculate the fraction of net precipitation that is stored
            precip_store = calculate_precip_store(S, precip_net, x1)

            # Calculate the amount of evaporation from storage
            evap_store = calculate_evap_store(S, evap_net, x1)

            # Update the storage by adding effective precipitation and
            # removing evaporation
            S = S - evap_store + precip_store
            # After updating, S still cannot be larger than x1, either smaller than 0
            S = jnp.clip(S, a_min=0, a_max=x1)

            # Update the storage again to reflect percolation out of the store
            perc = calculate_perc(S, x1)
            S = S - perc
            S = jnp.clip(S, a_min=0, a_max=x1)

            # The precip. for routing is the sum of the rainfall which
            # did not make it to storage and the percolation from the store
            current_runoff = perc + (precip_net - precip_store)

            # runoff_history keeps track of the recent runoff values in a vector
            # that is shifted by 1 element each timestep.
            runoff_history = jnp.roll(runoff_history, 1)
            runoff_history = index_update(runoff_history, index[0], current_runoff)
            # runoff_history = tt.set_subtensor(runoff_history[0], current_runoff)

            # here the unit hydrograph is not same as that in Xinanjiang model. No convolution here.
            Q9 = 0.9 * jnp.dot(runoff_history, UH1)
            Q1 = 0.1 * jnp.dot(runoff_history, UH2)

            # x3 must be positive
            x3 = jnp.clip(x3, a_min=0.0001)
            # R cannot exceed x3
            R = jnp.clip(R, a_min=0, a_max=x3)
            F = x2 * (R / x3) ** 3.5
            R = jnp.maximum(0, R + Q9 + F)
            # R cannot exceed x3
            R = jnp.clip(R, a_max=x3)

            Qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
            R = R - Qr

            Qd = jnp.maximum(0, Q1 + F)
            Q = Qr + Qd

            # The order of the returned values is important because it must correspond
            # up with the order of the kwarg list argument 'outputs_info' to lax.scan.
            return [S, runoff_history, R], Q

        # Move the time dimension to position 0 移动第二维到第一维，其他不变，所以就相当于第一维和第二维换位置了
        inputs = jnp.moveaxis(inputs, 1, 0)
        # apply_fun_scan is for streamflow_step; params are parameters
        f = partial(apply_fun_scan, params[1])
        # state_variables will be updated in each iteration in the loop
        # _ means state_variables_new; h_new is streamflow
        _, h_new = lax.scan(f, h, inputs)
        return h_new

    return init_fun, apply_fun


key = random.PRNGKey(1)
num_dims = 365  # Number of timesteps
batch_size = 1  # Batchsize

# Initialize the model
init_fun, gr4j = GR4J()
_, params = init_fun(key, (batch_size, num_dims, 1))


def mse_loss(parameters, inputs, targets):
    """ Calculate the Mean Squared Error Prediction Loss. """
    preds = gr4j(parameters, inputs)
    return jnp.mean((preds - targets) ** 2)


step_size = 1e-4
opt_init, opt_update, get_params = optimizers.sgd(step_size)
opt_state = opt_init(params)


@jit
def update(param, x, y, opt_state):
    """ Perform a forward pass, calculate the MSE & perform a SGD step. """
    loss, grads = value_and_grad(mse_loss)(param, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, loss


train_loss_log = []
start_time = time.time()
num_batches = 5
data = pd.read_csv(os.path.join(definitions.ROOT_DIR, "example", "sample.csv"))

# we should guarantee that when scaling, P E and Q are still in same level. Hence, we scale them together.
data_used_for_scale = data.values[:, [0, 1, -1]].T.reshape(-1, 1)
data_scale = pp.MinMaxScaler().fit_transform(data_used_for_scale)  # default range is [0, 1]
P_scale = data_scale[:data.shape[0], 0]
E_scale = data_scale[data.shape[0]:2 * data.shape[0], 0]
Q_scale = data_scale[2 * data.shape[0]:3 * data.shape[0], 0]

P = jnp.array(P_scale)
E = jnp.array(E_scale)
Q = jnp.array(Q_scale)

for batch_idx in range(num_batches):
    x_in = jnp.array(
        [P[batch_idx:batch_idx + num_dims], E[batch_idx:batch_idx + num_dims]])
    y = jnp.array(Q[batch_idx:batch_idx + num_dims])
    params, opt_state, loss = update(params, x_in, y, opt_state)
    batch_time = time.time() - start_time
    train_loss_log.append(loss)

    start_time = time.time()
    print("Batch {} | T: {:0.2f} | MSE: {:0.6f} | states: {}| params: {}".format(batch_idx, batch_time, loss, params[0],
                                                                                 params[1]))
