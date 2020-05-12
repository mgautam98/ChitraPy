from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ChitraPy",
    version="0.0.2",
    author="Gautam Mishra",
    author_email="mishragautam96@gmail.com",
    description="Digital Image Processing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgautam98/ImgPy",
    packages=find_packages(exclude=['tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)