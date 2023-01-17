import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

VERSION = "0.0.1"
DESCRIPTION = "Data pipeline for electrode grid recordings"
LONG_DESCRIPTION = "A package that provides easy access to electrode grid recordings and useful processing methods."

# Setting up
setup(
    name="gridtools",
    version=VERSION,
    author="weygoldt (Patrick Weygoldt)",
    author_email="<weygoldt@pm.me>",
    url="https://github.com/weygoldt/gridtools",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    py_modules=["gridtools", "utils"],
    packages=find_packages(),
    install_requires=requirements,
    keywords=["efish", "neurobiology", "neuroethology"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Biologists",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={"console_scripts": ["datacleaner = gridtools.datacleaner_cli:main"]},
    include_package_data=True,
)
