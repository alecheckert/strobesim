#!/usr/bin/env python
"""
setup.py

"""
import setuptools

setuptools.setup(
    name="strobesim",
    version="1.0",
    packages=setuptools.find_packages(),
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="simple single particle tracking simulations in 2D imaging geometries"
)
