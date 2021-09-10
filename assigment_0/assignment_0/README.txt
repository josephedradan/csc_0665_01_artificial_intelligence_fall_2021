Joseph Edradan 920783419

For the first question, the solution is trivial, just put a plus sign between a and b and return it.
For the second question, you need to know about generators to optimally get the lowest memory usage and the fastest time. Also, you need to know that the sum function takes in iterables.
Basically, you just do the sum of an iterator which is faster than the sum of a iterable because python will optimize that operation in C.
For the third question, you need to know about the min function and the optional kwargs you can add. The "key" kwarg allows you to apply your own sorting function.
