自建的RL算法库。
# TODO
1. load_model函数设计不合理，文件名格式不统一。
2. 加入tensorboard的功能。
3. 解决内存泄漏急需。(完成)
# 更新信息
## 2021年8月1日
1. 添加了PPO算法。
2. 是添加多线程worker(目前仅支持Linux仅使用CPU运算)。
3. 优化运行速度。
## 2021年8月2日
1. 添加setup.py。
## 2021年8月3日
1. 规定optimize函数返回值。
### 2021年8月15日
1. 解决内存泄漏问题。(方式较为粗暴)
### 2021年8月20日
1. 增加reward scale。
2. 添加全新的参数配置方式。

# 安装说明
`git clone https://github.com/yangtao121/RL
`

`git checkout package`

`pip install -e .`