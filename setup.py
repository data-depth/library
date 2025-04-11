from setuptools.command.build_ext import build_ext
from setuptools import Extension, setup, find_packages
import glob
import sys

class custom_build_ext(build_ext):
    def build_extensions(self):
        # Override the compiler executables. Importantly, this
        # removes the "default" compiler flags that would
        # otherwise get passed on to to the compiler, i.e.,
        # distutils.sysconfig.get_var("CFLAGS").
        self.compiler.set_executable("compiler_so", "g++ -fpic")
        self.compiler.set_executable("compiler_cxx", "g++")
        if sys.platform=='darwin':
        	self.compiler.set_executable("linker_so", "g++ -Wl, -shared -lstdc++")
        if sys.platform=='linux':
        	self.compiler.set_executable("linker_so", "g++ -Wl,--gc-sections -shared -lstdc++")
        build_ext.build_extensions(self)

if sys.platform=='darwin' or sys.platform=='linux':
	setup(
	    name="data_depth",
	    version="1.0.1",
	    author="Pavlo Mozharovskyi",
	    author_email="pavlo.mozharovskyi@telecom-paris.fr",
	    description="The package provides many procedures for calculating the depth of points in an empirical distribution for many notions of data depth",
	    long_description="The package provides many procedures for calculating the depth of points in an empirical distribution for many notions of data depth",
	    long_description_content_type="text/markdown",
	    packages=find_packages(),
	    install_requires=['numpy','scipy','scikit-learn'],
	    include_package_data=True,
	    ext_modules=[
		Extension(
		    "ddalpha", 
		    sources=["depth/src/ddalpha.cpp"],
		    extra_compile_args=["-I.",'-std=c++14','-fPIC','-O2'],
		    extra_link_args=["-rdynamic",'-std=c++14','-fPIC']
		),Extension(
		    "depth_wrapper", 
		    sources=["depth/src/depth_wrapper.cpp"],
		    extra_compile_args=["-I.",'-std=c++14','-fPIC','-O2'],
		    extra_link_args=["-rdynamic",'-std=c++14','-fPIC']
		)
	     
	   	
	    ],
	    data_files=[('depth/multivariate', glob.glob("depth/multivariate/*.rst"))],
	    zip_safe=False,
	    cmdclass={"build_ext": custom_build_ext}
	)
if sys.platform=='win32':
	setup(
		name="data_depth",
	        version="1.0.1",
	        author="Pavlo Mozharovskyi",
	        author_email="pavlo.mozharovskyi@telecom-paris.fr",
	        description="Procedures for calculating data depth in empirical distributions",
	        long_description="The package provides many procedures for calculating the depth of points in an empirical distribution for many notions of data depth",
	        long_description_content_type="text/markdown",
	        packages=find_packages(),
	        install_requires=['numpy', 'scipy', 'scikit-learn'],
	        include_package_data=True,
	        package_data={
	            "depth": ["src/*.dll"],  # <-- ceci inclura les DLL dans la wheel
	        },
	        zip_safe=False
	)
	
