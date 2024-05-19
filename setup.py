#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2023-10-28 09:16:46
LastEditTime: 2024-03-28 12:07:42
LastEditors: Wenyu Ouyang
Description: The setup script
FilePath: \hydromodel\setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import io
import pathlib
from os import path as op
from setuptools import setup, find_packages

readme = pathlib.Path("README.md").read_text(encoding='utf-8')
here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Wenyu Ouyang",
    author_email="wenyuouyang@outlook.com",
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="hydrological models starting from XinAnJiang",
    entry_points={
        "console_scripts": [
            "hydromodel=hydromodel.cli:main",
        ],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="hydromodel",
    name="hydromodel",
    packages=find_packages(include=["hydromodel", "hydromodel.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OuyangWenyu/hydromodel",
    version='0.2.3',
    zip_safe=False,
)
