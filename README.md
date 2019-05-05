# hydro-model-xaj
新安江水文模型

#### 项目介绍
**正在开发中**......

根据中国水利水电出版社出版的，由武汉大学叶守泽老师和河海大学詹道江老师等合编的《工程水文学》第三版教材中新安江模型的原理，
结合河海大学芮孝芳老师《水文学原理》中的相关知识，并参考以往学校前辈的程序，改编的新安江三水源模型的python版本。

#### 主要内容
1. 数据处理：

   采用txt格式作为首选，适合nc格式的地方也尝试使用nc数据格式，比如有经纬度的时间序列数据——雷达测雨数据等。原因可以参考：
[Python programming guide for Earth Scientists](http://python.hydrology-amsterdam.nl/manuals/hydro_python_manual.pdf)。
nc格式数据python接口官方资料：[netcdf4-python](https://github.com/Unidata/netcdf4-python)，
补充一些资料如下：[Generating NetCDF files with Python](http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/11_create_netcdf_python.pdf)；
[Python-NetCDF reading and writing example with plotting](http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html)。

2. 新安江模型核心算法
- 在进行产流计算之前，首先需要计算前期土壤蓄水容量。利用前期降雨蒸发数据，调用流域蒸发计算模型计算。

3. 模型参数率定


#### 使用说明
首先需要创建自己所需的输入数据文件input.xlsx，按照上面的描述，填好自己的数据。
然后直接运行每个模型文件夹下的主函数即可

#### 参与贡献

1. Fork 本项目
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request