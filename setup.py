from setuptools import setup

setup(
    name='aitkens',
    version='0.0.0',
    packages=['aitkens'],
    url='https://github.com/jftsang/aitkens',
    license='CC BY 4.0',
    author='J. M. F. Tsang',
    author_email='j.m.f.tsang@cantab.net',
    description='Aitken\'s delta-squared acceleration',
    install_requires=['numpy'],
    extras_require={'develop': ['hypothesis']},
)
