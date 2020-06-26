# DQN解决MountainCar

作业要求：得分大于-140

## Model的搭建

- 两层隐层（128，ReLu），输出层因为需要输出Q值所以size是act_dim，并且没有激活函数（因为需要负的Q值）,1000 episodes出头就可以轻松达到-140以上
- 尝试过多添加一层（64，ReLu），没有什么变化
- Learning Rate 选取0.005效果不错就没再试
