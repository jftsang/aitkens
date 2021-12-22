# aitkens

**Aitken's delta-squared series acceleration method**

J. M. F. Tsang (j.m.f.tsang@cantab.net)

---

## Usage

Given an object `xs` that can be turned into a one-dimensional numpy 
array, run:

```python
from aitkens import accelerate

accelerate(xs)
```

### Example: Iterates of $\sqrt{2}$

This example, which is given on the Wikipedia article [1], is actually a
poor example since the original iterates converge quadratically, rather 
than linearly. The accelerated sequence's terms tend to overshoot the
true values.

```python
from itertools import accumulate
from aitkens import accelerate

iterates = list(accumulate(
    range(5), lambda x, _: 0.5 * (x + 2/x), initial=1
))
acc = accelerate(iterates)
```

## References

  1. https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process

