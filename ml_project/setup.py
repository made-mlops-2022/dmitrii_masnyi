from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="my_module",
    packages=find_packages(),
    version="0.1.0",
    description="HW1 Mlops",
    author="Abovecat",
    entry_points={
        "console_scripts": ["ml_example_train = my_module.train:train_pipeline_command"]
    },
    install_requires=required,
    license="MIT",
)
