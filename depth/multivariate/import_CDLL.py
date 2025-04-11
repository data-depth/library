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

    if sys.platform == 'win32':
        site_packages = [p for p in sys.path if 'site-packages' in p or 'dist-packages' in p]
        for i in site_packages:
            print(f"Checking path: {i}")  # Ajout d'une impression pour le débogage
            os.add_dll_directory(i)
            ddalpha_exact = glob.glob(i + '/depth/src/*ddalpha*.dll')
            ddalpha_approx = glob.glob(i + '/depth/src/*depth_wrapper*.dll')
            print(f"Found DLLs: {ddalpha_exact + ddalpha_approx}")  # Ajout d'une impression pour le débogage
            if ddalpha_exact + ddalpha_approx != []:
                libExact = ct.CDLL(r"" + ddalpha_exact[0])
                libApprox = ct.CDLL(r"" + ddalpha_approx[0])
                print("DLLs loaded successfully.")
            else:
                print("No DLLs found.")
                libExact,libApprox=0,0

    return libExact,libApprox

libExact,libApprox=import_CDLL()
