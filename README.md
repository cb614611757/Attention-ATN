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

![adv_loss.png](adv_loss.png "image1")
![perturb_loss.png](perturb_loss.png "image2")
<div align=center><img src="total_loss.png"/></div>

Compare with raw images and adversarial Examples:

![raw image1](cam_image/image1.png "百度图片")
![Adversarial Example1](cam_image/1.png "百度图片")

![raw image2](cam_image/image2.png "百度图片")
![Adversarial Example2](cam_image/2.png "百度图片")

![raw image3](cam_image/image4.png "百度图片")
![Adversarial Example3](cam_image/3.png "百度图片")

<img width="150" height="150" src="cam_image/image1.png"/>
<img width="150" height="150" src="cam_image/1.png"/>
<img width="150" height="150" src="cam_image/image2.png"/>
<img width="150" height="150" src="cam_image/2.png"/>
<img width="150" height="150" src="cam_image/image4.png"/>
<img width="150" height="150" src="cam_image/3.png"/>


<center class="half">
    <img src="cam_image/image1.png" width="300"/><img rc="cam_image/1.png" width="300"/>
</center>

<center class="half">
    <img src="cam_image/image2.png" width="300"/><img src="cam_image/2.png" width="300"/>
</center>

<center class="half">
    <img src="cam_image/image4.png" width="300"/><img src="cam_image/3.png" width="300"/>
</center>

<center class="half">
<div align=left><img width="300" height="300" src="cam_image/image1.png"/></div>    <div align=center><img width="300" height="300" src="cam_image/1.png"/></div>
</center>

<center class="half">
<div align=left><img width="300" height="300" src="cam_image/image2.png"/></div>    <div align=center><img width="300" height="300" src="cam_image/2.png"/></div>
</center>

<center class="half">
<div align=left><img width="300" height="300" src="cam_image/image4.png"/></div>    <div align=center><img width="300" height="300" src="cam_image/3.png"/></div>
</center>

