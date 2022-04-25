# hydro-model-xaj

## What is hydro-model-xaj

Hydro-model-xaj is a python implementation for XinAnJiang (XAJ) model, which is one of the most famous conceptual
hydrological model, especially in Southern China.

**Not official version, just for learning** (Because objective condition of authors engineering level and urgent time,
errors may exist)

## How to run

Hydro-model-xaj is a Python console program (no graphic interface now). It is **still developing**, and we have not
provided a pip or conda package for hydro-model-xaj yet, so please set up python environment for the code.

If you are new to python,
Please [install miniconda or anaconda](https://github.com/waterDLut/WaterResources/blob/master/tools/jupyterlab&markdown.md#12-jupyterlab%E5%90%AF%E5%8A%A8)
.

Since you see hydro-model-xaj in GitHub, I think you have known a little about git and GitHub at least. If not, you can
see [this](https://github.com/waterDLut/WaterResources/blob/master/tools/git%26github.md#1-git%E7%9A%84%E5%AE%89%E8%A3%85)
to install git and register your own GitHub account.

Then, fork hydro-model-xaj to your GitHub, and clone it to your local computer (Linux or Windows).

If you have forked it before, please
see [this](https://github.com/waterDLut/WaterResources/blob/doc/tools/git%26github.md#55-fork%E5%90%8E%E5%90%8C%E6%AD%A5%E6%BA%90%E7%9A%84%E6%96%B0%E6%9B%B4%E6%96%B0%E5%86%85%E5%AE%B9)
to update it from [upstream](https://github.com/OuyangWenyu/hydro-model-xaj) as our previous version has some errors.

Open you terminal, then input：

```Shell
# clone hydro-model-xaj, if you have cloned it, ignore this step 
git clone <address of hydro-model-xaj in your github>
# move to it
cd hydro-model-xaj
# if updating from upstream, pull the new version to local
git pull
# create python environment
conda env create -f environment.yml
# activate it
conda activate xaj
```

Then you can run the following code to try hydro-model-xaj:

```Shell
cd hydromodel/app
python calibrate_xaj.py
```

To use your own data to run the model, we set a data interface, here is the convention:

- All input data for models are three-dimensional numpy array: [time, basin, variable], which means "time" series data
  for "variables" in "basins"
- Data files should be .npy files with a json file which show the information of the data. We provide sample code in
  "test/test_data.py" to show how to process your .csv/.txt file to the required format and load your reformatted data
- The name of .npy file and .json file must be "basins_lump_p_pe_q.npy" and "data_info.json". An example could be seen
  in the "example" directory
- You can set your own dataset when using scripts in the "app" directory like the following code:

```Shell
python calibrate_xaj.py --data_dir D:/code/hydro-model-xaj/hydromodel/example
```

## Why does hydro-model-xaj exists

When we want to learn about rainfall-runoff process and make forecast for flood, etc. We often use classic hydrological
models such as XAJ as baseline because it is trusted by many engineers and researchers. However, after searching in the
website very few repositories could be found. One day I happened to start learning Python, so I decided to implement the
model with Python. Previous commits for hydro-model-xaj have some errors, but now at least one executable version is
provided.

Actually open source science has brought great impact on hydrological modeling. For example, SWAT and VIC are very
popular now as they are public with great performance and readable documents; as more and more people use them, they
become more stable and powerful. XAJ is a nice model used by many engineers for practical production. We need to inherit
and develop it. I think hydro-model-xaj is a good start.

## What are the main features

We basically implement the formula in this book
-- [《流域水文模拟》](https://xueshu.baidu.com/usercenter/paper/show?paperid=ad9c545a7baa43321db97f5f16d393bf&site=xueshu_se)

Other reference Chinese books：

- ["*Principles of
  Hydrology*"/《水文学原理》](https://xueshu.baidu.com/usercenter/paper/show?paperid=5b2d0a40e2d2804f47346ae6ccf2d142&site=xueshu_se)
- ["*Hydrologic
  Forecasting*"/《水文预报》](https://xueshu.baidu.com/usercenter/paper/show?paperid=852a9a90a7d26c5fae749169f87b61e0&site=xueshu_se)
- ["*Engineering
  Hydrology*"/《工程水文学》](https://xueshu.baidu.com/usercenter/paper/show?paperid=6e2d38726c8e3c0b9f3a14bafb156481&site=xueshu_se)

More English references could be seen in the end of this README file.

The model mainly include three parts:

![](docs/source/img/xaj.jpg)

For the first part, we use an evaporation coefficient K (ratio of potential evapotranspiration to reference crop
evaporation generally from Allen, 1998) rather than Kc (the ratio of potential evapotranspiration to pan evaporation)
because we often use potential evapotranspiration data from system like GLDAS, NLDAS, etc. But it doesn't matter, when
you use pan evaporation, just treat K as Kc.

For the second part, we provide multiple implementations, because for this module, formula in different books are a
little different. One simplest version is chosen as default setting. More details could be seen in the document which
will be provided soon (You can see details in the source code directly now).

For the third part -- routing module, we provide different ways: the default is a common way with recession constant (
CS) and lag time (L) shown in the figure; second (You can set model's name as "xaj_mz" to use it) is a model
from [mizuRoute](http://www.geosci-model-dev.net/9/2223/2016/) to generate unit hydrograph for surface runoff (Rs -> Qs)
, as its parameters are easier to set, and we can optimize all parameters in a uniform way.

We provide two common calibration methods to optimize XAJ's parameters:

- [SCE-UA](https://doi.org/10.1029/91WR02985) from [spotpy](https://github.com/thouska/spotpy)
- [GA](https://en.wikipedia.org/wiki/Genetic_algorithm) from [DEAP](https://github.com/DEAP/deap): now only the method
  is used, but no completed case is provided yet. We will provide one soon.

Now the model is only for **one computing element** (typically, a headwater catchment). Soon we will provide calibration
for multiple headwater catchments. To get better simulation for large basins, a (semi-)distributed version may be
needed, and it is not implemented yet. The following links may be useful:

- https://github.com/ecoon/watershed-workflow
- https://github.com/ConnectedSystems/Streamfall.jl

Other implementations for XAJ:

- Matlab: https://github.com/wknoben/MARRMoT/blob/master/MARRMoT/Models/Model%20files/m_28_xinanjiang_12p_4s.m
- Java: https://github.com/wfxr/xaj-hydrological-model
- R, C++: https://github.com/Sibada/XAJ

## How to contribute

If you want to add features for hydro-model-xaj, for example, write a distributed version for XAJ, please create a new
git branch for your feature and send me a pull request.

If you find any problems in hydro-model-xaj, please post your questions
on [issues](https://github.com/OuyangWenyu/hydro-model-xaj/issues).

## References

- Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998. Crop Evapotranspiration, Food and Agriculture Organization of
  the United Nations, Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
- Duan, Q., Sorooshian, S., and Gupta, V. (1992), Effective and efficient global optimization for conceptual
  rainfall-runoff models, Water Resour. Res., 28( 4), 1015– 1031, doi:10.1029/91WR02985.
- François-Michel De Rainville, Félix-Antoine Fortin, Marc-André Gardner, Marc Parizeau, and Christian Gagné. 2012.
  DEAP: a python framework for evolutionary algorithms. In Proceedings of the 14th annual conference companion on
  Genetic and evolutionary computation (GECCO '12). Association for Computing Machinery, New York, NY, USA, 85–92.
  DOI:https://doi.org/10.1145/2330784.2330799
- Houska T, Kraft P, Chamorro-Chavez A, Breuer L (2015) SPOTting Model Parameters Using a Ready-Made Python Package.
  PLoS ONE 10(12): e0145180. https://doi.org/10.1371/journal.pone.0145180
- Mizukami, N., Clark, M. P., Sampson, K., Nijssen, B., Mao, Y., McMillan, H., Viger, R. J., Markstrom, S. L., Hay, L.
  E., Woods, R., Arnold, J. R., and Brekke, L. D.: mizuRoute version 1: a river network routing tool for a continental
  domain water resources applications, Geosci. Model Dev., 9, 2223–2238, https://doi.org/10.5194/gmd-9-2223-2016, 2016.
- Zhao, R.J., Zhuang, Y. L., Fang, L. R., Liu, X. R., Zhang, Q. S. (ed) (1980) The Xinanjiang model, Hydrological
  Forecasting Proc., Oxford Symp., IAHS Publication, Wallingford, U.K.
- Zhao, R.J., 1992. The xinanjiang model applied in China. J Hydrol 135 (1–4), 371–381.