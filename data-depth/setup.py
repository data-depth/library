from setuptools import setup, find_packages, Extension

depthCpp = Extension('depthCpp',
                     define_macros = [('MAJOR_VERSION', '0'),
                                      ('MINOR_VERSION', '1')],
                     include_dirs = ['src/data-depth/HDEst'],
                     sources = ['src/data-depth/HDEst/APD.cpp',
                                'src/data-depth/HDEst/auxLinAlg.cpp',
                                'src/data-depth/HDEst/HD.cpp',
                                'src/data-depth/HDEst/Matrix.cpp',
                                'src/data-depth/HDEst/MD.cpp',
                                'src/data-depth/HDEst/mvrandom.cpp',
                                'src/data-depth/HDEst/PD.cpp',
                                'src/data-depth/HDEst/ProjectionDepths.cpp',
                                'src/data-depth/HDEst/ZD.cpp'],
                     extra_compile_args=['-std=c++14'])

setup(
    name="data-depth",
    version="0.0.1",
    author="Pavlo Mozharovskyi, Rainer Dyckerhoff, Romain Valla",
    author_email="pavlo.mozharovskyi@telecom-paris.fr",
    description="A python library for data depth",
    long_description="A python library for data depth",
    long_description_content_type="text/markdown",
    url="https://data-depth.github.io",
    project_urls={
        "Bug Tracker": "https://data-depth/library",
        "Bug Tracker": "https://data-depth/data-depth.github.io",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    ext_modules = [depthCpp]
)
