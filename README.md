* Learning MNIST by mxnet

MINIST data file format is binary.
So, I wanted to use mxnet record format.

mnist_png is the project changing mnist binary file
to png file.
https://github.com/myleott/mnist_png

I made ctl file to read png, and published im2rec command and
make record file. color option will make grayscale record file.

```sh
im2rec training.ctl  ./ training.rec encoding=.png color=0
im2rec testing.ctl  ./ testing.rec encoding=.png color=0
```

* execute MNIST
published perl image.pl but test is not passed.
