""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings
This code solve generalized eigenvalue problem for operators or matrices"""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn

def lanczos_generalized(
    operator,
    metric_operator=None,
    metric_inv_operator=None,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_gpu=False,
):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        return operator.apply(x.float()).cpu().numpy()
    scipy_op = ScipyLinearOperator(shape, _scipy_apply)

    if isinstance(metric_operator, np.ndarray) or \
       isinstance(metric_operator, ScipyLinearOperator):
        metric_op = metric_operator
    else:
        def _scipy_apply_metric(x):
            x = torch.from_numpy(x)
            if use_gpu:
                x = x.cuda()
            return metric_operator.apply(x.float()).cpu().numpy()
        metric_op = ScipyLinearOperator(shape, _scipy_apply_metric)

    if isinstance(metric_inv_operator, np.ndarray) or \
       isinstance(metric_inv_operator, ScipyLinearOperator):
        metric_inv_op = metric_inv_operator
    else:
        def _scipy_apply_metric_inv(x):
            x = torch.from_numpy(x)
            if use_gpu:
                x = x.cuda()
            return metric_inv_operator.apply(x.float()).cpu().numpy()
        metric_inv_op = ScipyLinearOperator(shape, _scipy_apply_metric_inv)

    if init_vec is None:
        init_vec = np.random.rand(size)
    elif isinstance(init_vec, torch.Tensor):
        init_vec = init_vec.cpu().numpy()
    eigenvals, eigenvecs = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        M=metric_op,
        Minv=metric_inv_op,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T
