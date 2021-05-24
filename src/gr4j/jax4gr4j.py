from functools import partial
import jax.numpy as jnp
from jax import lax
from jax.ops import index, index_add, index_update


def calculate_precip_store(S, precip_net, x1):
    """Calculates the amount of rainfall which enters the storage reservoir."""
    n = x1 * (1 - (S / x1) ** 2) * jnp.tanh(precip_net / x1)
    d = 1 + (S / x1) * jnp.tanh(precip_net / x1)
    return n / d


# Determines the evaporation loss from the production store
def calculate_evap_store(S, evap_net, x1):
    """Calculates the amount of evaporation out of the storage reservoir."""
    n = S * (2 - S / x1) * jnp.tanh(evap_net / x1)
    d = 1 + (1 - S / x1) * jnp.tanh(evap_net / x1)
    return n / d


# Determines how much water percolates out of the production store to streamflow
def calculate_perc(current_store, x1):
    """Calculates the percolation from the storage reservoir into streamflow."""
    return current_store * (1 - (1 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25)


def hydrograms(x4_limit, x4):
    """Produces a vector which partitions an input of water into streamflow over
    the course of several days.
    This function is intended to be run once as a part of GR4J, before the main
    loop executes.
    Parameters
    ----------
    x4_limit : scalar Theano tensor
        An upper limit on the value of x4, which is real-valued
    x4 : scalar Theano tensor
        A parameter controlling the rapidity of transmission of stream inputs
        into streamflow.
    Returns
    -------
    UH1 : 1D Theano tensor
        Partition vector for the portion of streamflow which is routed with a
        routing store.
    UH2 : 1D Theano tensor
        Partition vector for the portion of streamflow which is NOT routed with
        the routing store.
    """
    timesteps = jnp.arange(2 * x4_limit)
    SH1 = jnp.where(timesteps <= x4, (timesteps / x4) ** 2.5, 1.0)
    SH2A = jnp.where(timesteps <= x4, 0.5 * (timesteps / x4) ** 2.5, 0)
    SH2B = jnp.where((x4 < timesteps) & (timesteps <= 2 * x4),
                     1 - 0.5 * (2 - timesteps / x4) ** 2.5, 0)

    # The next step requires taking a fractional power and
    # an error will be thrown if SH2B_term is negative.
    # Thus, we use only the positive part of it.
    SH2B_term = jnp.maximum((2 - timesteps / x4), 0)
    SH2B = jnp.where((x4 < timesteps) & (timesteps <= 2 * x4),
                     1 - 0.5 * SH2B_term ** 2.5, 0)
    SH2C = jnp.where(2 * x4 < timesteps, 1, 0)

    SH2 = SH2A + SH2B + SH2C
    UH1 = SH1[1::] - SH1[0:-1]
    UH2 = SH2[1::] - SH2[0:-1]
    return UH1, UH2


def streamflow_step(params, state_variables, forcings):
    """Logic for simulating a single timestep of streamflow from GR4J within Jax.
    This function is usually used as an argument to lax.scan as the inner function for a loop.

    Parameters
    ----------
    params :
        x1 : scalar tensor
            Storage reservoir parameter;
        x2 : scalar tensor
            Catchment water exchange parameter;
        x3 : scalar tensor
            Routing reservoir parameters;
        UH1 : 1D tensor
            Partition vector routing daily stream inputs into multiday streamflow
            for the fraction of water which interacts with the routing reservoir;
        UH2 : 1D tensor.
            Partition vector routing daily stream inputs into multiday streamflow
            for the fraction of water which does not interact with the routing reservoir.
    state_variables :
        S: scalar Theano tensor
            Beginning value of storage in the storage reservoir;
        runoff_history : 1D Theano tensor
            Previous days' levels of streamflow input. Needed for routing streamflow;
            over multiple days.
        R : scalar Theano tensor
            Beginning value of storage in the routing reservoir.
    forcings :
        P : scalar tensor
            Current timestep's value for precipitation input.
        E : scalar tensor
            Current timestep's value for evapotranspiration input.

    Returns
    -------
    S : scalar tensor
        Storage reservoir level at the end of the timestep
    runoff_history : 1D tensor
        Past timesteps' stream input values
    R : scalar tensor
        Routing reservoir level at the end of the timestep
    Q : scalar tensor
        Resulting streamflow
    """
    # params
    x1 = params[0]
    x2 = params[1]
    x3 = params[2]
    UH1 = params[3]
    UH2 = params[4]
    # state_variables
    S = state_variables[0]
    runoff_history = state_variables[1]
    R = state_variables[2]
    # forcings
    P = forcings[0]
    E = forcings[1]
    # Calculate net precipitation and evapotranspiration
    precip_difference = P - E
    precip_net = jnp.maximum(precip_difference, 0)
    evap_net = jnp.maximum(-precip_difference, 0)

    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S, precip_net, x1)

    # Calculate the amount of evaporation from storage
    evap_store = calculate_evap_store(S, evap_net, x1)

    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store

    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S, x1)
    S = S - perc

    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + (precip_net - precip_store)

    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = jnp.roll(runoff_history, 1)
    runoff_history = index_update(runoff_history, index[0], current_runoff)
    # runoff_history = tt.set_subtensor(runoff_history[0], current_runoff)

    Q9 = 0.9 * jnp.dot(runoff_history, UH1)
    Q1 = 0.1 * jnp.dot(runoff_history, UH2)

    F = x2 * (R / x3) ** 3.5
    R = jnp.maximum(0, R + Q9 + F)

    Qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
    R = R - Qr

    Qd = jnp.maximum(0, Q1 + F)
    Q = Qr + Qd

    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to lax.scan.
    return [S, runoff_history, R], Q


def simulate_streamflow(P, E,
                        S0, Pr0, R0, x1, x2, x3, x4, x4_limit):
    """Simulates streamflow over time using the model logic from GR4J as implemented in Jax.
    This function can be used in jax-numpy-based libraries to
    offer up the functionality of GR4J with added gradient information.
    Parameters
    ----------
    P : 1D tensor
      Time series of precipitation
    E : 1D tensor
      Time series of evapotranspiration
    S0 : scalar tensor
      Initial value of storage in the storage reservoir.
    Pr0 : 1D tensor
      Initial levels of streamflow input. Needed for routing streamflow.
      If this is nonzero, then it is implied that there is initially
      some streamflow which must be routed in the first few timesteps.
    R0 : Initial tensor
      Beginning value of storage in the routing reservoir.
    x1 : scalar tensor or 1D tensor
      Storage reservoir parameter
    x2 : scalar tensor
      Catchment water exchange parameter
    x3 : scalar tensor
      Routing reservoir parameters
    x4 : scalar tensor
      Routing time parameter
    tv_x1 : boolean
      Determines whether or not x1 is allowed to vary over time, i.e. whether
      x1 is a scalar or a vector.
    Returns
    -------
    streamflow : 1D tensor
      Time series of simulated streamflow
    """
    UH1, UH2 = hydrograms(x4_limit, x4)
    # sequence-first variables are needed for loop calculation
    forcings = jnp.moveaxis(jnp.array([P, E]), 1, 0)
    parameters = [x1, x2, x3, UH1, UH2]
    state_variables = [S0, Pr0, R0]

    f = partial(streamflow_step, parameters)
    # state_variables will be updated in each iteration in the loop
    state_variables_new, streamflow = lax.scan(f, state_variables, forcings)
    return streamflow
