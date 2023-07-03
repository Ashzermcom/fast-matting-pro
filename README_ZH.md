# Fast Matting Pro
Fast Matting 是一个通用的抠图框架，集成了优秀的SAM抠图算法

![image](docs/demo.gif)
# What's New
2023-07-03 新增webui交互界面，支持点，框，文本提示，用于辅助抠图。

# Installation

# Quick Start
[快速开始](docs/GET_START.md)

# In the future
当前版本的模型文本提示对模型效果的影响不强，主要原因是只是简单将clip提取的特征，混入sparse prompt。增强图文匹配和特征提取的版本将在未来上线。

# 致谢
## 参考项目
[fast-reid](https://github.com/JDAI-CV/fast-reid) \
[sam](https://github.com/facebookresearch/segment-anything)
```
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}

@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
## 开发人员
WebUI: 方前
Algorithm: Asher
