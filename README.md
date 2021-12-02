自建的RL算法库。
# 依赖项
tensorflow>=2.6.0rc1
# 更新信息
## 2021年8月1日
1. 添加了PPO算法。
2. 是添加多线程worker(目前仅支持Linux仅使用CPU运算)。
3. 优化运行速度。
## 2021年8月2日
1. 添加setup.py。
## 2021年8月3日
1. 添加新的并行计算方法，可以支持部分需要环境并行的系统。
2. 针对bullet这类环境需要定制PPO，参考PPO_Multi。
## 2021年8月5日
1. 内存泄漏解决方法：使用tensorflow2.6.0rc2即可。

# This project has been migrated to https://gitee.com/yangtaohome/RL
