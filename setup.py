import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ChitraPy",
    version="0.0.1",
    author="Gautam Mishra",
    author_email="mishragautam96@gmail.com",
    description="Digital Image Processing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgautam98/ImgPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)