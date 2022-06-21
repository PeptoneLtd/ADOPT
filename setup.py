# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools
from setuptools import setup

from adopt import __version__

with open("README.md") as f:
    readme = f.read()

setup(
    name="adopt",
    version=__version__,
    packages=setuptools.find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/PeptoneInc/ADOPT",
    license="MIT",
    author="Peptone Ltd.",
    author_email="carlo@peptone.io",
    description=" Attention based DisOrder PredicTor (adopt): intrinsic protein disorder prediction through"
    "deep bidirectional transformers",
    data_files=[(".", ["LICENSE", "README.md", "CHANGELOG.md", "CITATION.cff"])],
    zip_safe=True,
)