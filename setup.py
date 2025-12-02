from setuptools import setup, find_packages

setup(
    name="quasarpy",
    version="0.1.0",
    description="Python wrapper for ODYSSEE CAE Quasar Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ben Thornton",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.0.0",
    ],
    package_data={
        "quasarpy": ["qsr_scripts/*.qsr"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.7",
)
