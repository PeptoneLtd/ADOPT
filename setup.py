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
    zip_safe=True,
)
