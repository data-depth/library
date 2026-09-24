import numpy as np



def isHermitian(H,tol=1e-10):
    """
    Parameters
    ----------
        H  : {array-like} of shape (d,d).
            Matrix to check if it is hermitian.
         
        tol  : float
            tolerance for hermitian check. 

    Returns :
    ---------
        boolena to determine if it is hermitian or not
    """
    
    return np.allclose(H,H.conj().T, atol=tol)

def isHPD(H, tol=1e-10):
    """Check Hermitian positive-definiteness."""
    if H.shape[0] != H.shape[1]:
        return False
    if not isHermitian(H, tol):
        return False
    try:
        np.linalg.cholesky(H)
        return True
    except np.linalg.LinAlgError:
        return False