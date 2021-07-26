# hydro-model-xaj

## 项目介绍

根据中国水利水电出版社出版的，由武汉大学叶守泽老师和河海大学詹道江老师等合编的《工程水文学》第三版教材中新安江模型的原理，结合河海大学芮孝芳老师《水文学原理》中的相关知识，并重点参考了河海大学包为民老师的《水文预报》第5版， 编写的 **三水源新安江模型** 的 python 版本。**非官方版本，仅供学习参考**。

## 环境配置

本项目在python环境下运行，首先[安装miniconda或者anaconda](https://github.com/waterDLut/WaterResources/blob/master/tools/jupyterlab&markdown.md#12-jupyterlab%E5%90%AF%E5%8A%A8)。

然后fork本项目到自己的github下，想要把github上的项目拉到本地运行，最好还是自己在本地[安装好git工具](https://github.com/waterDLut/WaterResources/blob/master/tools/git%26github.md#1-git%E7%9A%84%E5%AE%89%E8%A3%85)，这样便于自己的版本控制。

在自己本地想要放置本项目的地方，打开终端，输入：

```Shell
# clone项目
git clone <本项目在你的github中的地址>
# 进入本项目
cd hydro-model-xaj
# 创建项目python环境
conda env create -f environment.yml
# 激活项目python环境
conda activate xaj
```

## 使用说明

**注：本项目正在维护中，相应计划请看[这里](https://github.com/OuyangWenyu/hydro-model-xaj/projects/1)**

自己以调试方式运行 test 文件夹下的 python 程序即可查看具体的代码运行。

目前测试函数包括直接调用模型的测试函数，以及率定的测试函数，简单修改程序即可测试任意一个。

测试使用的数据较少，并不符合实际预报规范，目前项目只是为了梳理新安江模型原理及其运算过程，实际应用还在本项目代码基础上进一步开发。

如果觉得项目中文字和代码对原理的理解有误，或者出现代码运行错误，请在[issues](https://github.com/OuyangWenyu/hydro-model-xaj/issues)中留言。
