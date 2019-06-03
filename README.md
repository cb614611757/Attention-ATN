Attention-ATN
======
Tensorflow implementation of Attention-ATN: A Method to Generate Transferable Adversarial Examples}

USAGE

Download the weight about defense model,cycle_gan model and based-model used to calculate cam-matrix from
[google clouds](https://drive.google.com/drive/folders/1iYP53cRqVhfXXY_eYN-spWFrvhPFD67N "x")

Download the data sets:[AAAC-2019 data sets](https://tianchi.aliyun.com/competition/entrance/231701/information "x")
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

Compare with raw images and adversarial examples:

<table>
    <tr>
        <td ><center><img width="150" height="150" src="cam_image/image1.png" >image1 </center></td>
        <td ><center><img width="150" height="150" src="cam_image/cam1.png" >Grad_cam1 </center></td>
        <td ><center><img width="150" height="150" src="cam_image/1.png"  >adversarial example1 </center></td>
    </tr>
</table>

<table>
    <tr>
        <td><center><img width="150" height="150" src="cam_image/image2.png"  >image2 </center></td>
        <td ><center><img width="150" height="150" src="cam_image/cam2.png" >Grad_cam2 </center></td>
        <td ><center><img width="150" height="150" src="cam_image/2.png"  >adversarial example2 </center> </td>
    </tr>
</table>

<table>
    <tr>
        <td><center><img width="150" height="150" src="cam_image/image4.png"   >image3 </center></td>
        <td ><center><img width="150" height="150" src="cam_image/cam3.png" >Grad_cam3 </center></td>
        <td><center><img width="150" height="150" src="cam_image/3.png"  >adversarial example3 </center></td>
    </tr>

</table>



