import sys

import setuptools


# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""


setuptools.setup(
    name="aux-ssm-samplers",
    author="Adrien Corenflos",
    description="Auxiliary samplers for state-space models with tractable densities",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "jaxtyping>=0.2.8",
    ],
    long_description_content_type="text/markdown",
    keywords="probabilistic state space bayesian statistics sampling algorithms",
    license="MIT",
    license_files=("LICENSE",),
)