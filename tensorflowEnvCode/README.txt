CS510 A6: Tensorflow Tuning
Bradley Pospeck and Brandon Perry

Python version: 3.5.5
tensorflow/tensorboard version: 1.8.0

Currently the code is set to run with the following parameter values:
	learning rate: .001
	dropout: 0.9
	use_two_fc: False (which means only using one fc layer)
	use_two_conv: True (which means using 2 conv layers)
These can be changed in 'main()' or added to; Just edit the lists in the for loops.
These for loops allow the code to be ran with multiple architectures in 1 go.

The 'labels_1024.tsv' and 'sprite_1024.png' are used in the code. 
They should be in the same directory as the python file when running.

Code ran on a Windows 10 machine. 
Warnings do come up when running the code and tensorboard (as with the class examples),
but it has always run successfully.

Code base used is adapted from the following sources:
	https://github.com/decentralion/tf-dev-summit-tensorboard-tutorial
	https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py

