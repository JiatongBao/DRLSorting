# Learn Multi-Step Object Sorting Tasks through Deep Reinforcement Learning

This repository will provide PyTorch code for training and testing object sorting policies with deep reinforcement learning in both simulation and real-world settings on a UR5/UR3 robot arm. This is the reference implementation for the manuscript submitted to <i>Robotica</i>.

### Learn Multi-Step Object Sorting Tasks through Deep Reinforcement Learning

Robotic systems are traditionally controlled to repetitively perform specific actions for manufacturing tasks. These control methods are usually domain-dependent and model-dependent and cost lots of human efforts. They cannot meet the emerging requirements of generality and flexibility in many areas such as intelligent manufacturing and customized production. This paper develops a general model-free approach to enable robots to perform multi-step object sorting tasks through deep reinforcement learning. Taking projected heightmap images from different time steps as input without extra high-level image analysis and understanding, critic models are designed for producing a pixel-wise Q value map for each type of action. It is a new trial to apply pixel-wise Q value based critic networks on solving multi-step sorting tasks that involve many types of actions and complex action constraints. The experimental validations on simulated and realistic object sorting tasks demonstrate the effectiveness of the proposed approach.

<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img src="images/method.png" width="60%"/></div>

#### Demo Videos

These two videos show how the simulated robot performs on sorting 6 cuboid blocks with random colors after 5000-step and 30000-step trial-and-errors, respectively.
<table>
<tr>
<td><img src="images/sort_six_blocks_5000.gif"></td>
<td><img src="images/sort_six_blocks_30000.gif"></td>
</tr>
<tr>
<td align="center">after 5,000 steps</td>
<td align="center">after 30,000 steps</td>
</tr>
</table>

In order to show the potential capability of our model, the task of sorting 10 blocks with random colors and shapes is performed.
<table>
<tr>
<td><img src="images/sort_ten_blocks_30000.gif"></td>
<td><img src="images/sort_ten_blocks_50000.gif"></td>
</tr>
<tr>
<td align="center">after 30,000 steps</td>
<td align="center">after 50,000 steps</td>
</tr>
</table>

Demo videos of a real robot in action will be available.

#### Contact
If you have any questions or find any bugs, please let me know: [Jiatong Bao] jtbao[at]yzu[dot]edu[dot]cn

## Installation
To be continued.

#### Pretrained Models
Model1 (trained on tasks of sorting 4 blocks with fixed colors) 
[Download](https://drive.google.com/file/d/1_tFZJUNs0p9UkGV4AiDwF7955unbNRTS/view?usp=sharing)<br>
Model2 (trained on tasks of sorting 4 blocks with random colors) [Download](
https://drive.google.com/file/d/12Czid2KE0FcPVsUPCM_L4qba5_3qdnku/view?usp=sharing)<br>
Model3 (trained on tasks of sorting 6 blocks with random colors) [Download](
https://drive.google.com/file/d/1cdqIqTWBgmWhSkWcDLdIbAWwhDlH8Rfv/view?usp=sharing)