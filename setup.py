from setuptools import setup, find_packages

setup(
    name="dnaty",
    version="5.1.0",
    description="Dynamic Neuro-Adaptive sYstem with evoluTionarY Learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="dNaty Authors",
    url="https://github.com/pedrovergueiro/dNaty",
    packages=find_packages(exclude=["tests*", "web*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["matplotlib>=3.7.0", "jupyter", "pytest>=8.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="neural architecture search neuroevolution continual learning pytorch",
)
