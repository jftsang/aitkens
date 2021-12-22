import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='aitkens',
    version='0.1.0',
    packages=['aitkens'],
    url='https://github.com/jftsang/aitkens',
    license='CC BY 4.0',
    author='J. M. F. Tsang',
    author_email='j.m.f.tsang@cantab.net',
    description='Aitken\'s delta-squared acceleration',
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=['numpy'],
    extras_require={'develop': ['hypothesis']},
)
