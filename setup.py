import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "mrange",
    version = "1.1",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "mrange is used to remove nested range loops",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/mrange/archive/refs/tags/v1.1.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numba", "numpy"],
    python_requires=">=3.6",
)
