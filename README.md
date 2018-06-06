# NMF Neural Nets

The NMF optimizer doesn't work perfectly. But, the NMF would get well. 


## Results of MNIST

Now it cannot run well. Please see [test_optimizers.py](https://github.com/ashigirl96/sakurai-nmf/blob/master/sakurai_nmf/tests/test_optimizers.py) or [mnist.py](https://github.com/ashigirl96/sakurai-nmf/blob/master/sakurai_nmf/examples/mnist.py).

But now we can get accuracy 81% for only 5 iterations without ReLU.

```bash
python sakurai_nmf/examples/mnist.py --batch_size 5000 --use_bias --num_mf_iters 5 --num_bp_iters 0
Using TensorFlow backend.
NMF-optimizer
(5/5) [Train]loss 0.606, accuracy 89.420 time, 9.197 [Test]loss 0.715, accuracy 81.312
```

Use only Nonlinear semi-NMF.

```bash
$ python sakurai_nmf/examples/mnist.py --batch_size 5000 --use_relu --use_bias --num_mf_iters 5 --num_bp_iters 0
Using TensorFlow backend.
NMF-optimizer
(5/5) [Train]loss 0.602, accuracy 91.480 time, 10.648 [Test]loss 0.669, accuracy 87.272
```


## Results of Fashion MNIST


we can get accuracy 74% for only 3 iterations.


```bash
$ python sakurai_nmf/examples/mnist.py --batch_size 5000 --use_bias --num_mf_iters 5 --num_bp_iters 0 --dataset fashion
Using TensorFlow backend.
NMF-optimizer
(5/5) [Train]loss 0.579, accuracy 87.920 time, 9.588 [Test]loss 0.684, accuracy 78.3846
```
