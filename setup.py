from setuptools import setup, find_packages

setup(
    name="jacobi_package",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy"],
    description="метод якоби для собственных чисел и векторов",
    python_requires=">=3.8",
)