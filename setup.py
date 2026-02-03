import re
from setuptools import setup, find_packages

# Read version from __init__.py without importing the package
with open('quasarpy/__init__.py') as f:
    version = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read()).group(1)

setup(
    name="quasarpy",
    version=version,
    description="Python wrapper for ODYSSEE CAE Quasar Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ben Thornton",
    author_email="bthorn191@gmail.com",
    url="https://github.com/bthornton191/quasarpy",
    project_urls={
        "Bug Tracker": "https://github.com/bthornton191/quasarpy/issues",
        "Source Code": "https://github.com/bthornton191/quasarpy",
    },
    license="MIT",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.0.0",
        "plotly>=5.0.0",
        "ipywidgets>=7.0.0",
        "pymoo>=0.6.0",
        "tqdm>=4.0.0",
        "numba",

    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.0.0",
        ],
    },
    package_data={
        "quasarpy": ["qsr_scripts/*.qsr"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
)
