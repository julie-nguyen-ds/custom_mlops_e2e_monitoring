from setuptools import find_packages, setup

setup(
    name="telkomsel_home_credit_risk",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    description="Repository implementing an end-to-end MLOps workflow on Databricks for home credit risk detection.",
    authors="",
)
