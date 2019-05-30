Attention-ATN
======
Tensorflow implementation of ttention-ATN: A Method to Generate Transferable Adversarial \\
Examples}

USAGE

Download the weight about defense model,cycle_gan model and based-model used to calculate cam-matrix from
[http://www.baidu.com](http://www.baidu.com "x")

------
First run:

pip install -r requirements.txt

which will extract the MNIST dataset using the Keras API and train a simple CNN model that will serve as the 'target model' for the generator to trick.

------
Next, run:

python AdvGAN.py
This script will first train the generator. You can specifiy whether or not you want it to be targeted. A different generator will be trained for each target. You will want to tweak the weight paths for each target (or I will update that soon).

Once the training process is complete, the function attack will be called. This function will load the weights from the generator and run an attack on the test set. It will also print out a before and after picture of two images from the last batch.
