from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='imbaqu',
    version='1.0.0',
    author='Jelke Wibbeke',
    author_email='jelke.wibbeke@offis.de',
    description='A python package for the quantification of data imbalance.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/OFFIS-ROC/imbaqu',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'KDEpy'],
    include_package_data= True
)
