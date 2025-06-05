import ctypes as ct
import multiprocessing as mp 
import sys, os, glob

def import_CDLL():
    if sys.platform=='linux':
        for i in sys.path :         
            if i.split('/')[-1]=='site-packages' or i.split('/')[-1]=='dist-packages':#Add search dist-packages
                de=glob.glob(i+'/*ddalpha*.so')
                da=glob.glob(i+'/*depth_wrapper*.so')
                if da!=[] and de!=[]:
                    ddalpha_exact=glob.glob(i+'/*ddalpha*.so')
                    ddalpha_approx=glob.glob(i+'/*depth_wrapper*.so')
        libExact=ct.CDLL(ddalpha_exact[0])
        libApprox=ct.CDLL(ddalpha_approx[0])
        
    if sys.platform=='darwin':
        for i in sys.path :
            if i.split('/')[-1]=='site-packages' or i.split('/')[-1]=='dist-packages':#Add search dist-packages
                de=glob.glob(i+'/*ddalpha*.so')
                da=glob.glob(i+'/*depth_wrapper*.so')
                if da!=[] and de!=[]:
                    ddalpha_exact=glob.glob(i+'/*ddalpha*.so')
                    ddalpha_approx=glob.glob(i+'/*depth_wrapper*.so')
        libExact=ct.CDLL(ddalpha_exact[0])
        libApprox=ct.CDLL(ddalpha_approx[0])

    if sys.platform=='win32':
        site_packages = [p for p in sys.path if ('site-packages' in p) or ("dist-packages" in p)] #Add search dist-packages 
        for i in site_packages:
            dll_path = os.path.join(i, 'depth', 'src')
            if os.path.isdir(dll_path): #check it is a real dir
                os.add_dll_directory(dll_path)
                ddalpha_exact=glob.glob(os.path.join(dll_path, '*ddalpha*.dll'))
                ddalpha_approx=glob.glob(os.path.join(dll_path, '*depth_wrapper*.dll'))
                if ddalpha_exact+ddalpha_approx!=[]:
                    libExact=ct.CDLL(r""+ddalpha_exact[0])
                    libApprox=ct.CDLL(r""+ddalpha_approx[0])

    return libExact,libApprox

libExact,libApprox=import_CDLL()
