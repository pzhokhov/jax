import jax
import jax.numpy as np
import numpy as onp
from jax import grad, jit, vmap
from jax import random
from jax import tree_util as tu

from mpi4py import MPI
from functools import wraps, partial

from jax.core import Primitive
from jax.interpreters.ad import deflinear

_mpi_allmean_primitives = {}
_mpi_bcast_primitives = {}

def _mpi_allmean(x, comm=MPI.COMM_WORLD):
    return x
    x_np = onp.asarray(x)
    recv = onp.zeros_like(x_np)
    comm.Allreduce(x_np, recv, op=MPI.SUM)
    return recv / comm.size


def _mpi_bcast_transpose(x, root=0, comm=MPI.COMM_WORLD):
    '''
    backpropagate through bcast - allreduce for root rank, zero for the other ranks
    '''
    x_np = onp.asarray(x)
    recv = comm.reduce(x_np, root=root, op=MPI.SUM)
    if recv is None:
        recv = onp.zeros_like(x_np)
    return recv


def make_allmean(comm=None):
    '''
    construct differentiable allmean operation with a given communicator
    '''
    comm = comm or MPI.COMM_WORLD
    key = id(comm)
    if key not in _mpi_allmean_primitives:
        prim = Primitive(f'mpi_allmean_{comm}')
        prim.def_impl(lambda x: _mpi_allmean(x, comm=comm))
        prim.def_abstract_eval(lambda x: x)
        deflinear(prim, lambda x: [prim.bind(x)])
        _mpi_allmean_primitives[key] = lambda x: prim.bind(x)
        jax.interpreters.xla.translations[prim] = 'pyfunc'
    return _mpi_allmean_primitives[key]


def make_bcast(comm=None, root=0):
    '''
    construct differentiable bcast operation with a given communicator and root rank
    '''
    comm = comm or MPI.COMM_WORLD
    key = (id(comm), root)
    if key not in _mpi_bcast_primitives:
        prim = Primitive(f'mpi_bcast_{comm}')
        prim.def_impl(lambda x: comm.bcast(x, root=root))
        prim.def_abstract_eval(lambda x: x)
        deflinear(prim, lambda x: [_mpi_bcast_transpose(x, root, comm)])
        _mpi_bcast_primitives[key] = lambda x: prim.bind(x)
    return _mpi_bcast_primitives[key]


def mpi_optimizer(comm=None):
    '''
    Decorator to create a mpi-wrapped version of an optimizer
    '''
    def _mpi_optimizer_inner(opt_maker):
        @wraps(opt_maker)
        def wrapped_opt_maker(*args, **kwargs):
            return mpi_wrap(opt_maker(*args, **kwargs), comm=comm)
        return wrapped_opt_maker
    return _mpi_optimizer_inner


def mpi_wrap(init_update_pair, comm=None, jit_update_fun=True):
    '''
    make an mpi version of optimizer - sync
    the states at the beginning of the training,
    and run mean-allreduce on gradients on each update
    '''
    _comm = comm or MPI.COMM_WORLD
    init_fun, update_fun = init_update_pair
    if jit_update_fun:
        update_fun = jax.jit(update_fun)
    allmean = make_allmean(comm)
    bcast = make_bcast(comm)
    # sync_from_root = partial(mpi_sync_from_root, comm=_comm)
    # allmean = mpi_allmean

    @wraps(init_fun)
    def mpi_init_fun(x0):
        x0 = bcast(x0)
        return init_fun(x0)

    @wraps(update_fun)
    def mpi_update_fun(i, g, *state):
        flat_g, shapes, treedef = flatten_numpy_tree(g)
        flat_g = allmean(flat_g)
        g = unflatten_numpy_tree(flat_g, shapes, treedef)
        # todo - periodic check if synced / re-sync?
        return update_fun(i, g, *state)

    return mpi_init_fun, mpi_update_fun


def flatten_numpy_tree(tree):
    '''
    Utility function that converts tree of numpy array objects
    into a single vector (and extracts shapes of elements and a treedef)
    '''
    flat, treedef = tu.tree_flatten(tree)
    shapes = [d.shape for d in flat]
    flatflat = np.concatenate([d.flatten() for d in flat])
    return flatflat, shapes, treedef


def unflatten_numpy_tree(flatflat, shapes, treedef):
    '''
    Utiliy function that converts flat vector, shapes of each element in a tree and a treedef
    into a object tree of numpy arrays
    '''
    idx = 0
    flat = []
    for s in shapes:
        l = int(np.prod(s))
        flat.append(flatflat[idx:idx + l].reshape(s))
        idx += l
    return tu.tree_unflatten(treedef, flat)


mpi_opt = mpi_optimizer()
