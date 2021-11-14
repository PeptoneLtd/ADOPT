# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
import setuptools

with open("adopt/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

setup(
    name='adopt',
    version=version,
    packages=setuptools.find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/PeptoneInc/ADOPT',
    license='MIT',
    author='Peptone Ltd.',
    author_email='kamil@peptone.io',
    description=' Attention DisOrder PredicTor (adopt): intrinsic protein disorder prediction throughdeep bidirectional transformers',
    data_files=[(".", ["LICENSE", "README.md", "CHANGELOG.md", "CITATION.cff"])],
    zip_safe=True,
)


