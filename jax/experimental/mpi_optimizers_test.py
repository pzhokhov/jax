from jax import grad,jit,disable_jit
import jax.numpy as np
import jax.tree_util as tu
from jax.experimental.optimizers import sgd, adam, rmsprop
from mpijax.with_mpi import with_mpi
from mpijax.optimizers import mpi_wrap, flatten_numpy_tree, unflatten_numpy_tree, make_allmean, make_bcast
from mpi4py import MPI
import numpy as onp
import pytest
from functools import partial


@pytest.mark.parametrize('opt_maker', [sgd, adam, rmsprop])
# @with_mpi(2)
def test_mpi_vs_nonmpi(opt_maker):
    '''
    make sure results of mpi optimizer match non-mpi version
    '''
    comm = MPI.COMM_WORLD

    def loss(x, data):
        return np.dot(x + data, x + data) / np.prod(data.shape)
    data = [onp.random.normal(size=(5,)) for _ in range(comm.size)]
    data_local = comm.scatter(data)
    data = onp.array(data).flatten()
    data = comm.bcast(data)
    x0 = np.ones(1, dtype=np.float32)
    niters = 10
    init_fun, update_fun = mpi_wrap(opt_maker(step_size=0.01), jit_update_fun=True)
    x = init_fun(x0)
    states = []
    for i in range(niters):
        states.append(x)
        g = grad(loss)(x[0], data)
        x = jit(update_fun)(i, g, x)
    x = init_fun(x0)
    states_local = []
    for i in range(niters):
        states_local.append(x)
        g = grad(loss)(x[0], data_local)
        x = update_fun(i, g, x)
    if comm.rank == 0:
        for state_nompi, state_mpi in zip(states, states_local):
            onp.testing.assert_allclose(state_nompi[0], state_mpi[0])


@pytest.mark.parametrize('opt_maker', [sgd, adam, rmsprop])
@with_mpi(2)
def test_update_differentiable(opt_maker):
    step_size = 0.01
    init_fun, update_fun = mpi_wrap(opt_maker(step_size=step_size))
    x0 = np.zeros((2,), dtype=np.float32)
    g0 = np.zeros_like(x0)
    x0 = init_fun(x0)
    
    @jit
    def sum_updates(g): return sum(update_fun(0, g, x0)[0])
    dsum_updates = grad(sum_updates)(g0)
    if opt_maker == sgd:
        onp.testing.assert_allclose(
            dsum_updates, -np.ones_like(g0) * step_size)

@with_mpi(2)
def test_mpimean_differentiable():
    comm = MPI.COMM_WORLD
    allmean = jit(make_allmean(comm))
    def L(x): return sum(allmean(x)) * (comm.rank + 1)
    x = np.zeros((1,))
    gradL = jit(grad(L))(x)
    onp.testing.assert_allclose(gradL, (comm.size + 1) / 2)


@with_mpi(4)
def test_bcast():
    comm = MPI.COMM_WORLD
    root = 1
    bcast = make_bcast(comm, root=root)
    x0 = np.ones((3,)) * (comm.rank + 1)
    test = bcast(x0)
    onp.testing.assert_allclose(2 * np.ones_like(x0), test)
    testgrad = grad(lambda x: sum(bcast(x)))(x0)
    if comm.rank == root:
        onp.testing.assert_allclose(np.ones_like(x0) * comm.size, testgrad)
    else:
        onp.testing.assert_allclose(np.zeros_like(x0), testgrad)


def test_numpy_flatten_unflatten():
    tree = _build_numpy_tree()
    reconstructed = unflatten_numpy_tree(*flatten_numpy_tree(tree))
    tu.tree_multimap(onp.testing.assert_allclose, tree, reconstructed)


def _build_numpy_tree():
    return {
        'a': onp.ones((10,)),
        'b': [onp.zeros((5,)), onp.random.normal(size=(2,))]
    }


if __name__ == '__main__':
    test_mpimean_differentiable()
