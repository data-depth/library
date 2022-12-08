from setuptools.command.build_ext import build_ext
from setuptools import Extension, setup, find_packages

import os, sys
import platform
import glob



setup(
    	name="depth",
   	version="1.0.0",
    	author="Pavlo mozharovskyi",
    	author_email="pavlo.mozharovskyi@telecom-paris.fr",
   	description="The package provides many procedures for calculating the depth of points in an empirical distribution for many notions of data depth",
   	long_description="The package provides many procedures for calculating the depth of points in an empirical distribution for many notions of data depth",
   	long_description_content_type="text/markdown",
   	url="https://github.com/pypa",
   	platforms='linux',
   	install_requires=['numpy','scipy','scikit-learn'],
    	packages=find_packages(),
    	ext_modules=[
        
   	 ],
   	
   	include_package_data=True,
   	
   	#data_files=[('depth', ['depth/Win64/ddalpha.dll'])],
   	data_files=[('depth/Win64', glob.glob("depth/Win64/*.dll")),('depth/Win32', glob.glob("depth/Win32/*.dll")),('depth/UNIX', glob.glob("depth/UNIX/*.so")),('depth/MACOS', glob.glob("depth/MACOS/*.so"))],
	

   	zip_safe=False,
	)






	
