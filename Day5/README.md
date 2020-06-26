# DDPG 解决四轴飞行器

## ActorModel

```Python
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################
        #offset
        self.fc1 = layers.fc(size=64, act='relu')
        self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=64, act='relu')
        self.fc4 = layers.fc(size=64, act='relu')
        self.fc5 = layers.fc(act_dim, act='tanh')
        #main power
        self.pfc1 = layers.fc(size=64, act='relu')
        self.pfc2 = layers.fc(size=64, act='relu')
        self.pfc3 = layers.fc(size=64, act='relu')
        self.pfc4 = layers.fc(size=64, act='relu')
        self.pfc5 = layers.fc(1, act='tanh')
        self.final = layers.fc(act_dim, act='tanh')
        ######################################################################
        ######################################################################

    def policy(self, obs):
        ######################################################################
        ######################################################################
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h4 = self.fc4(h3)
        h5 = self.fc5(h4) * 0.2
        ph1 = self.pfc1(obs)
        ph2 = self.pfc2(ph1)
        ph3 = self.pfc3(ph2)
        ph4 = self.pfc4(ph3)
        ph5 = self.pfc5(ph4)
        logits = fluid.layers.elementwise_add(h5, ph5)
        logits = self.final(logits)     
        

        ######################################################################
        ######################################################################
        return logits
```

#### 注：

1. ActorModel输入obs，输出action
2. 使用一个main power控制四轴的统一电压，然后用四个offset（*0.2）去微调
3. 输出层使用tanh使得输出在-1到1之间
4. 尝试过增加隐层大小，效果不好，反而更难训练

## CriticModel

```Python
class CriticModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        self.fc1 = layers.fc(size=64, act='relu')
        self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=64, act='relu')
        self.fc4 = layers.fc(size=64, act='relu')
        self.fc5 = layers.fc(size=1, act=None)
        ######################################################################
        ######################################################################

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        ######################################################################
        ######################################################################
        concat = layers.concat([obs, act], axis=1)
        h1 = self.fc1(concat)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h4 = self.fc4(h3)
        Q = self.fc5(h4)
        Q = layers.squeeze(Q, axes=[1])
        ######################################################################
        ######################################################################
        return Q
```

#### 注：

1. CriticModel输入obs和action，输出一个Q分数
2. 先concatenate obs和action
3. 隐层为64，ReLu
4. 输出层为Q值，所以不用激活函数
5. 尝试过增加隐层大小，效果不好，反而更难训练

## 参数

参考：https://harikrishnansuresh.github.io/assets/deep-rl-final.pdf

```Python
######################################################################
######################################################################
#
# 1. 请设定 learning rate，尝试增减查看效果
#
######################################################################
######################################################################
ACTOR_LR = 0.0005   # Actor网络更新的 learning rate 
d_alr=0.000001
CRITIC_LR = 0.005   # Critic网络更新的 learning rate 
d_clr=0.00001

GAMMA = 0.98        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.01         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward
```

### Learning Rate

- 初始actor和critic分别为0.0005和0.005，另外设置d_alr和d_clr为0.000001和0.00001
- 当step超过TRAIN_TOTAL_STEPS/8时，两个lr缩小为之前的1/2.5倍 （0.0002，0.002）
- 当step超过TRAIN_TOTAL_STEPS/4时，两个lr缩小为之前的1/2倍 （0.0001，0.001）
- 当step超过TRAIN_TOTAL_STEPS/2时，两个lr缩小为之前的1/2倍 （0.00005，0.0005）
- 当step超过TRAIN_TOTAL_STEPS*3/4时，两个lr缩小为之前的1/2倍 （0.000025，0.00025）
- 如果分数超过当前最好值，分别减小一个d_alr和d_clr
- 如果分数超过8000，分别减小一个d_alr和d_clr
- 如果分数超过14000，分别减小两个d_alr和d_clr

## 结果

- 在二十五万个steps之后出现了最优模型分数超过八千在8900左右，这时通过可视化发现飞行器几乎可以停在初始位置左右，并且不会出现坠落的情况，不知道分数达到14000时会是怎样
- 继续训练至一百万个steps，发现分数开始止步不前，之后甚至有时波动为负数，原因不明，总之不完全收敛。通过可视化发现训练一百万次后的模型，飞行器不能停在原地，不过也不会坠毁。
