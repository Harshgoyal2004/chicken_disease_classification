from setuptools import setup, find_packages

setup(
    name="cnnClassifier",
    version="0.0.1",
    author="Harsh Goyal",
    description="A small package for chicken disease classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "PyYAML",
        "python-box==6.0.2",
        "matplotlib",
        "seaborn",
        "dvc",
        "scipy",
        "Flask",
        "Flask-Cors",
    ]
)
