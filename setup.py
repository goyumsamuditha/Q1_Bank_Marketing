from setuptools import setup, find_namespace_packages

setup(
    name="bank-term-deposit-mlops",
    version="0.1.1",
    description="Bank Term Deposit MLOps project",
    packages=find_namespace_packages(include=["src*"]),
)