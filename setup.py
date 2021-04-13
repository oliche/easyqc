import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setuptools.setup(
    name="easyqc",
    version="0.2.0",
    author="Olivier Winter",
    description="Seismic viewer for numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oliche/easyqc",
    project_urls={
        "Bug Tracker": "https://github.com/oliche/easyqc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=require,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    package_data={'easyqc': ['easyqc.ui', 'easyqc.svg']},
    python_requires=">=3.7",
)
