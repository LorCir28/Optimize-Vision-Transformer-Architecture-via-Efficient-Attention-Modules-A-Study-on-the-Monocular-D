# Project of Vision and Perception exam at La Sapienza University of Rome (Artificial intelligence and Robotics master degree)

# Optimize Vision Transformer architecture via efficient attention modules: a study on the monocular depth estimation task

Two modifications [1,2] of METER [3] attention module are proposed and implemented to build two its versions: Meta-METER and Pyra-METER.

The performances of the two novel architectures are tested on the Intel Xeon CPU GHz, Coral DevBoard equipped with ARM Cortex-A53, NVIDIA Jetson TX1 equipped with Cortex-A57.

Evaluation metrics:
- total inference time
- transformer block inference time
- RMSE, MAE, AbsRel and accuracy values


## References
[1] Yu, Weihao, et al. "Metaformer is actually what you need for vision." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[2] Wang, Wenhai, et al. "Pyramid vision transformer: A versatile backbone for dense prediction without convolutions." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[3] Papa, Lorenzo, Paolo Russo, and Irene Amerini. "METER: a mobile vision transformer architecture for monocular depth estimation." IEEE Transactions on Circuits and Systems for Video Technology (2023).




