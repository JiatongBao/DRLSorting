# Learn Multi-Step Object Sorting Tasks through Deep Reinforcement Learning

This repository will provide PyTorch code for training and testing object sorting policies with deep reinforcement learning in both simulation and real-world settings on a UR5/UR3 robot arm. This is the reference implementation for the manuscript submitted to Robotica:

### Learn Multi-Step Object Sorting Tasks through Deep Reinforcement Learning

Robotic systems are traditionally controlled to repetitively perform specific actions for manufacturing tasks. These control methods are usually domain-dependent and model-dependent and cost lots of human efforts. They cannot meet the emerging requirements of generality and flexibility in many areas such as intelligent manufacturing and customized production. This paper develops a general model-free approach to enable robots to perform multi-step object sorting tasks through deep reinforcement learning. Taking projected heightmap images from different time steps as input without extra high-level image analysis and understanding, critic models are designed for producing a pixel-wise Q value map for each type of action. It is a new trial to apply pixel-wise Q value based critic networks on solving multi-step sorting tasks that involve many types of actions and complex action constraints. The experimental validations on simulated and realistic object sorting tasks demonstrate the effectiveness of the proposed approach.

<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img src="images/method.png" width="60%"/></div>

#### Demo Videos

The video shows how the simulated robot performs on sorting 6 cuboid blocks with random colors after 30000 steps of trial-and-errors.
<div align="center"><img src="images/sort_six_blocks_30000.gif" width="60%"></div>

Demo videos of a real robot in action will be available.

#### Contact
If you have any questions or find any bugs, please let me know: [Jiatong Bao] jtbao[at]yzu[dot]edu[dot]cn

## Installation
To be continued.