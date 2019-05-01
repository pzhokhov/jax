import jax
import jax.test_util as jtu
import jax.numpy as np
from jax.core import Primitive
from jax.interpreters import ad, xla

class PyFuncTest(jtu.JaxTestCase):
  def testPyFunc(self):
    # the is really the example of how jax should not be used - a python function with a side effect
    call_counter = [0]
 
    @xla.pyfunc_linear('_my_f')   
    def f(x):
      call_counter[0] += 1
      return x

    g = jax.jit(lambda x: f(x) + 1)
    n_trials = 10
    call_counter_before = call_counter[0]
    for _ in range(n_trials):
      g(np.zeros((2,3), dtype=np.float32))
    call_counter_after = call_counter[0]
    assert call_counter_after == call_counter_before + n_trials

if __name__ == '__main__':
  PyFuncTest().testPyFunc()    
