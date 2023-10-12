#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

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
    python_requires='>= 3.6 ',  # Python版本要求
)
