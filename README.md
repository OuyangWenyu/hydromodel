# hydro-model-xaj
新安江水文模型

#### 项目介绍
**正在开发中**......

根据中国水利水电出版社出版的，由武汉大学叶守泽老师和河海大学詹道江老师等合编的《工程水文学》第三版教材中新安江模型的原理，
结合河海大学芮孝芳老师《水文学原理》中的相关知识，并参考以往学校前辈的程序，改编的新安江三水源模型的python版本。

#### 主要内容
1. 数据处理：

   暂时没有采取读写excel方式，原因可以参考：
[Python programming guide for Earth Scientists](http://python.hydrology-amsterdam.nl/manuals/hydro_python_manual.pdf)。
简而言之，就是我们很少有正版excel，即便有或者用wps等，excel也存在序列数据太长导致不得不分sheet等不方便的情况，
使得数据的获取inefficient，而使用python处理水文相关数据很好，再考虑以后类似项目数据的获取便利及尽可能地标准一些，
本项目尝试使用nc数据格式。nc格式数据python接口官方资料：[netcdf4-python](https://github.com/Unidata/netcdf4-python)，还有一些其他资料如下：
[Generating NetCDF files with Python](http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/11_create_netcdf_python.pdf)；
[Python-NetCDF reading and writing example with plotting](http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html)。

2. 新安江模型核心算法

3. 模型参数率定


#### 使用说明
首先需要创建自己所需的输入数据文件input.xlsx，按照上面的描述，填好自己的数据。
然后直接运行每个模型文件夹下的主函数即可

#### 参与贡献

1. Fork 本项目
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request