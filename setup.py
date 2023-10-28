"""
Author: Wenyu Ouyang
Date: 2023-10-28 09:16:46
LastEditTime: 2023-10-28 09:27:22
LastEditors: Wenyu Ouyang
Description: setup.py for hydromodel package
FilePath: \hydro-model-xaj\setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pathlib
from setuptools import setup, find_packages

readme = pathlib.Path("README.md").read_text()
setup(
    name="hydromodel",  # 输入项目名称
    version="0.0.1",  # 输入版本号
    keywords=[""],  # 输入关键词
    description="",  # 输入概述
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/iHeadWater/hydro-model-xaj",  # 输入项目Github仓库的链接
    author="iHeadWater",  # 输入作者名字
    author_email="",  # 输入作者邮箱
    license="MIT_license",  # 此为声明文件，一般填写 MIT_license
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[""],  # 输入项目所用的包
    python_requires=">= 3.7 ",  # Python版本要求
)
