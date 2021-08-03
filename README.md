自建的RL算法库。
# TODO
1. load_model函数设计不合理，文件名格式不统一。
# 更新信息
## 2021年8月1日
1. 添加了PPO算法。
2. 是添加多线程worker(目前仅支持Linux仅使用CPU运算)。
3. 优化运行速度。
## 2021年8月2日
1. 添加setup.py。
## 2021年8月3日
1. 规定optimize函数返回值。

# 安装说明
`git clone https://github.com/yangtao121/RL
`

`git checkout package`

`pip install -e .`