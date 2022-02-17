from mpi4py import MPI

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------------------------------------------
# small/temporary utility for saving/loading sparse matrices, plots...
# (should be cleaned !)

def source_name(source_type=None, source_proj=None):
    assert source_type and source_proj
    return source_type+'_'+source_proj

def rhs_fn(source_type, source_proj=None, eta=None, mu=None, nu=None, npz_suffix=True, prefix=True):
    if prefix:
        fn = 'rhs_'
    else:
        fn = ''
    fn += source_name(source_type, source_proj)
    if source_type == 'manu_J':
        assert (eta is not None) and (mu is not None) and (nu is not None)
        fn += '_eta'+repr(eta)+'_mu'+repr(mu)+'_nu'+repr(nu)
    if npz_suffix:
        fn += '.npz'
    return fn

def sol_ref_fn(source_type, N_diag, source_proj=None):
    fn = 'u_ref_'+source_name(source_type, source_proj)+'_N'+repr(N_diag)+'.npz'
    return fn

def hf_fn(hom_seq=True):  # domain_name):
    if hom_seq:
        fn = 'hf.npz'
    else:
        fn = 'hf_inhom.npz'
    return fn

def error_fn(source_type=None, method=None, conf_proj=None, k=None, domain_name=None,deg=None):
    return 'errors/error_'+domain_name+'_'+source_type+'_'+'_deg'+repr(deg)+'_'+get_method_name(method, k, conf_proj=conf_proj)+'.txt'

def get_method_name(method=None, k=None, conf_proj=None, penal_regime=None):
    if method == 'nitsche':
        method_name = method
        if k==1:
            method_name += '_SIP'
        elif k==-1:
            method_name += '_NIP'
        elif k==0:
            method_name += '_IIP'
        else:
            assert k is None
    elif method == 'conga':
        method_name = method
        if conf_proj is not None:
            method_name += '_'+conf_proj
    else:
        raise ValueError(method)
    if penal_regime is not None:
        method_name += '_pr'+repr(penal_regime)

    return method_name

def get_fem_name(method=None, k=None, DG_full=False, conf_proj=None, domain_name=None,nc=None,deg=None,hom_seq=True):
    assert domain_name
    fn = domain_name+(('_nc'+repr(nc)) if nc else '') +(('_deg'+repr(deg)) if deg else '')
    if DG_full:
        fn += '_fDG'
    if method is not None:
        fn += '_'+get_method_name(method, k, conf_proj)
    if not hom_seq:
        fn += '_inhom'
    return fn

def get_load_dir(method=None, DG_full=False, domain_name=None,nc=None,deg=None,data='matrices'):
    assert data in ['matrices','solutions','rhs']
    if method is None:
        assert data == 'rhs'
    fem_name = get_fem_name(domain_name=domain_name,method=method, nc=nc,deg=deg, DG_full=DG_full)
    return './saved_'+data+'/'+fem_name+'/'
