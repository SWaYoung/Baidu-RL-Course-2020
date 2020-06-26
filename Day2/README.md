# 使用Sarsa和Q-Learning走迷宫

## Sarsa 和 Q-Learning 的区别

- Sarsa是on policy，计算target Q的值时，使用Q(St+1,At+1)
- Q-learning是off policy，计算target Q的时候，选Q(St+1)的最大值
