# Policy Gradient解决Pong

作业要求：达到0分以上

## Model的搭建

- 只建了一层隐层(800, ReLu),输出层实际输出的是选择每一个action的可能性,所以是(act_dim, Softmax)
- Learning Rate 初始设置比较大0.0015, 1000 episodes之后减小为0.001，2000 episodes之后减小为0.0005
- 尝试过增加隐层效果不明显甚至有衰退
