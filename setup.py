import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "mrange",
    version = "0.6",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "test",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/mrangearchive/refs/tags/v0.6.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.6",
)
