from setuptools import setup, find_packages


setup(
    name="time-trace-templates",
    version="1.0.0",
    packages=find_packages(),
    author="Mart Pothast",
    author_email="mart.pothast@gmail.com",
    install_requires=[
        "numpy==1.22.3",
        "pandas==1.4.1",
        "matplotlib==3.5.1",
        "uproot==4.3.7",
        "iminuit==2.17.0",
        "numba==0.56.4",
        "scipy==1.9.3",
        "uncertainties==3.1.7",
        "scikit-learn==1.0.2",
    ],
    python_requires=">3.9",
)
