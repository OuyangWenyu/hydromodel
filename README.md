# hydro-model-xaj

新安江水文模型
**正在开发中**......

## 项目介绍

根据中国水利水电出版社出版的，由武汉大学叶守泽老师和河海大学詹道江老师等合编的《工程水文学》第三版教材中新安江模型的原理，结合河海大学芮孝芳老师《水文学原理》中的相关知识，编写的新安江三水源模型的 python 版本。

## 主要内容

### 数据处理：

采用 txt 格式作为首选，适合 nc 格式的地方也尝试使用 nc 数据格式，比如有经纬度的时间序列数据——雷达测雨数据等。原因可以参考：[Python programming guide for Earth Scientists](http://python.hydrology-amsterdam.nl/manuals/hydro_python_manual.pdf)。nc 格式数据 python 接口官方资料：[netcdf4-python](https://github.com/Unidata/netcdf4-python)，补充一些资料如下：[Generating NetCDF files with Python](http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/11_create_netcdf_python.pdf)；[Python-NetCDF reading and writing example with plotting](http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html)。

### 新安江模型核心算法

#### 前期土壤含水量计算

在进行产流计算之前，首先需要计算前期土壤蓄水容量。利用前期降雨蒸发数据，采用流域蒸发计算模型逐日计算。

- 当降雨 P 大于蒸发 E，即 P-E=PE>0。
  说明流域蓄水量增加，这部分增加的水量在流域蒸发模型中，首先被用来补充上层土壤水，在流域上层蓄水量达到最大值后，再补充下层，同理最后补充深层。在判断表层水是否补充充足时，需要计算产流量 R，产流量计算遵循蓄满产流原理（见下一小节）。在整个过程中，只有表层有蒸散发，且按蒸散发能力蒸发，K 是折算系数，EM 是实测的水面蒸发值，下层和深层为 0。
- 当 PE<0 时。
  说明流域失水，那么首先蒸发上层水，按上层蒸发能力蒸发，上层蒸发完之后再消耗下层的含水量，下层的蒸发量与下层含水量成正比，与蓄水容量成反比，在两层蒸发模型中根据剩余蒸发量，按比例直接计算可得。但是，**当下层含水量与下层蓄水容量之比小于深层蒸散发系数 C 时**，需要利用三层蒸发模型。

上面一段蒸发为什么这么计算，书上没有解释，_个人理解_：
下层含水量与下层蓄水容量之比小于深层蒸散发系数 C，深层蒸散发系数可以理解为深层蒸发量和深层蒸发能力之比，同样假设其等于深层含水量与深层蓄水容量之比，那么下层含水量与下层蓄水容量之比理解为下层蒸散发系数，也就是说下层蒸散发系数小于深层时，深层水应该承担一部分的蒸发量。但是如果下层也按深层的蒸散发系数蒸发，可以承担剩余的蒸发量，那就不需要深层水了。这里推理一下，应该是认为*下层水的蒸散发系数不是定值，看下层含水量情况以及和深层蒸散发系数比较的情况*，当 下层含水/下层蓄水容量 大于深层蒸散发系数时，深层就不必出力了，或者就算小，但是按照深层蒸散发系数蒸发也能满足除去表层蒸发之后的蒸发量时，那么也同样不需要深层含水了，如果不满足这两种情况，那就需要深层水分担一部分了。

最后汇总三层蒸发和三层蓄水。

整体计算流程图如下所示：

1. PE>=0 时

```flow
st=>start: 执行
sub=>subroutine: 计算R
cond2=>condition: WU+PE-R>WUM
op2=>operation: WU=WU+PE-R
cond3=>condition: WU+PE-R-WUM+WL>WLM
op3=>operation: WU=WUM,WL=WLM,WD=W+PE-R-WU-WL
op4=>operation: WU=WUM,WL=WU+WL+PE-R-WUM
op5=>operation: EU=K*EM,ED=EL=0
io=>inputoutput: E=EU+EL+ED,W=WU+WL+WD
e=>end: 结束

st->sub->cond2
cond2(yes)->cond3
cond2(no)->op2->op5
cond3(yes)->op3->op5
cond3(no)->op4->op5
op5->io->e
```

2. PE<0 时

```flow
st=>start: 执行
cond4=>condition: WU>PE
op7=>operation: EU=K*EM,EL=ED=0,WU=WU+PE
op12=>operation: .
op8=>operation: EU=WU+P,WU=0
cond5=>condition: WL>C*WLM
op9=>operation: EL=(K*EM-EU)*WL/WLM,WL=WL-EL,ED=0
cond6=>condition: WL>C*(K*EM-EU)
op10=>operation: EL=C*(K*EM-EU),WL=WL-EL,ED=0
op11=>operation: EL=WL,WL=0,ED=C*(K*EM-EU)-EL,WD=WD-ED
io=>inputoutput: E=EU+EL+ED,W=WU+WL+WD
e=>end: 结束

st->cond4
cond4(yes)->op7->op12->io
cond4(no)->op8->cond5
cond5(yes)->op9->io
cond5(no)->cond6
cond6(yes)->op10->io
cond6(no)->op11->io
```

#### 流域产流计算

产流计算基本思路：流域上每点含水量达到田间持水量时产流，但是各个点的田间持水量和初始水量都是不同的，如何表达？采用了一种概率方式，利用流域蓄水容量曲线表示。整个流域有初始土壤含水量 W0，其空间分布在蓄水容量曲线上进行表达；PE 的值及其分布同样可以在曲线上表达；那么产流量就可在图上表示出来。计算公式如下：

- P-E+a < $W_{mm^{'}}$时
  R = PE-$\int_a^{PE+a}[1-\varphi(W_{m^{'}})]dW_{m^{'}} $ = P - E - $(W_m-W_0)+W_m(1-\frac{a+PE}{W_{mm^{'}}})^{1+b}$
- 当 P-E+a $\geqq W_{mm^{'}}$时
  R=P-E-$(W_m-W_0)$

因此，需要先根据流域初始含水量求出 a 值。这里也是为什么要提前多天计算模型开始计算时间的流域土壤含水量的原因，因为直接在模型起算时间计算时，计算蒸发需要知道 R，而计算 R 需要知道初始土壤含水量，而初始含水量未知，a 无法求，产流量不可知，无法计算蒸发，会形成死循环。而从多天前开始起算，保证前期有降雨使土壤含水量达到过田间持水量，或者长期干旱，流域蓄水量基本为 0，都便于后面的计算。
在经过前面的计算，已知起算时间流域含水量的情况下，可以根据下式求得 a 值：
a = ${W_{mm^{'}}}[1-(1-\frac{W_0}{W_m}^{\frac{1}{1+b}})]$

（**有一个人 star 或者 fork 这个项目的话，我就把上面几个公式的推导过程写出来 ☺**）

#### 水源划分

水源划分的基本思路是产流面积上自由水的概率分布。

#### 坡地汇流

地下水线性水库模型。

#### 河网汇流

单位线以及河道演算。

### 模型参数率定

## 使用说明

直接运行test.py即可（**尚在开发中！！！**）

## 参与贡献

1. Fork 本项目
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request
