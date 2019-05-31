Attention-ATN
======
Tensorflow implementation of Attention-ATN: A Method to Generate Transferable Adversarial \\
Examples}

USAGE

Download the weight about defense model,cycle_gan model and based-model used to calculate cam-matrix from
[google clouds](https://drive.google.com/open?id=1bhQ43GSrG2JkiLgh4QedYPPrhgJd6rzG "x")

------
First run:

pip install -r requirements.txt

------
Next, run:

python train.py

After training some steps, the chenkpoints and attack images will be saved to local file holder.

The loss curve of training Attention-ATN:

![adv_loss.png](adv_loss.png "百度图片")
![perturb_loss.png](perturb_loss.png "百度图片")
![total_loss.png](total_loss.png "百度图片")

Compare with raw image and adversarial Examples:

![raw image1](cam_image/image1.png "百度图片")
![Adversarial Example1](cam_image/1.png "百度图片")

![raw image2](cam_image/image2.png "百度图片")
![Adversarial Example2](cam_image/2.png "百度图片")

![raw image3](cam_image/image4.png "百度图片")
![Adversarial Example3](cam_image/3.png "百度图片")


