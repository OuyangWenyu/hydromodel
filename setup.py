#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2022-06-28 15:06:30
LastEditTime: 2024-02-09 12:00:05
LastEditors: Wenyu Ouyang
Description: The setup script.
FilePath: \hydro-model-xaj\setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


import pathlib
from setuptools import setup, find_packages

readme = pathlib.Path("README.rst").read_text()
history = pathlib.Path("HISTORY.rst").read_text()
requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Wenyu Ouyang",
    author_email="wenyuouyang@outlook.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="hydromodel is a python implementation for common hydrological models such as the XinAnJiang (XAJ) model, which is one of the most famous conceptual hydrological models, especially in Southern China.",
    entry_points={
        "console_scripts": [
            "hydromodel=hydromodel.cli:main",
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="hydromodel",
    name="hydromodel",
    packages=find_packages(include=["hydromodel", "hydromodel.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OuyangWenyu/hydromodel",
    version="0.0.1",
    zip_safe=False,
)
