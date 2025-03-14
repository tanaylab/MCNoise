from setuptools import Extension, setup, find_packages

requirements = open("py_requirements.txt").read().split()

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    author="Dror Bar",
    author_email="dror.bar@weizmann.ac.il",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    description="Single-cell RNA Sequencing ambient noise removal",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    name="MCNoise",
    packages=find_packages(include=["MCNoise"]),
    url="https://github.com/tanaylab/MCNoise.git",
    version="0.1",
)