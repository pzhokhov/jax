from __future__ import print_function

from jax import grad, vjp, make_jaxpr
import jax.numpy as np

def f(x):
  return np.sin(np.sin(np.sin(np.sin(x))))


_, f_vjp = vjp(f, 3.)
print(make_jaxpr(f_vjp)(3.))
# { lambda e ;  ; a.
#   let b = pack * a
#       (c d) = id b
#       f = mul_any d e
#       g = pack f *
#       h = pack g
#   in h }


print(make_jaxpr(grad(f))(3.))
# { lambda b ;  ; a.
#   let c = pack a
#       (d) = id c
#       e = ones_like d
#       f = cos d
#       g = mul e f
#       h = sin d
#       i = cos h
#       j = mul g i
#       k = sin h
#       l = cos k
#       m = mul j l
#       n = sin k
#       o = cos n
#       p = mul m o
#       q = mul_any b p
#       r = pack q *
#   in r }

