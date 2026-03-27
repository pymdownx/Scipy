def _minimize_lbfgsb(fun, x0, args=(), jac=None, bounds=None,
                     disp=_NoValue, maxcor=10, ftol=2.2204460492503131e-09,
                     gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
                     iprint=_NoValue, callback=None, maxls=20,
                     finite_diff_rel_step=None, workers=None,
                     **unknown_options):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.

    Options
    -------
    disp : None or int
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.18.0.

    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``proj g_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations before minimization terminates.
        Note that this function may violate the limit if the gradients
        are evaluated by numerical differentiation.
    maxiter : int
        Maximum number of algorithm iterations.
    iprint : int, optional
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.18.0.

    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    workers : int, map-like callable, optional
        A map-like callable, such as `multiprocessing.Pool.map` for evaluating
        any numerical differentiation in parallel.
        This evaluation is carried out as ``workers(fun, iterable)``.

        .. versionadded:: 1.16.0

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.
    If the minimization is slow to converge the optimizer may halt if the
    total number of function evaluations exceeds `maxfun`, or the number of
    algorithm iterations has reached `maxiter` (whichever comes first). If
    this is the case then ``result.success=False``, and an appropriate
    error message is contained in ``result.message``.

    """
    _check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps

    x0 = asarray(x0).ravel()
    n, = x0.shape
    if disp is not _NoValue:
        warnings.warn("scipy.optimize: The `disp` and `iprint` options of the "
                      "L-BFGS-B solver are deprecated and will be removed in "
                      "SciPy 1.18.0.",
                      DeprecationWarning, stacklevel=3)

    if iprint is not _NoValue:
        warnings.warn("scipy.optimize: The `disp` and `iprint` options of the "
                      "L-BFGS-B solver are deprecated and will be removed in "
                      "SciPy 1.18.0.",
                      DeprecationWarning, stacklevel=3)

    # historically old-style bounds were/are expected by lbfgsb.
    # That's still the case but we'll deal with new-style from here on,
    # it's easier
    if bounds is None:
        pass
    elif len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    else:
        bounds = np.array(old_bound_to_new(bounds))

        # check bounds
        if (bounds[0] > bounds[1]).any():
            raise ValueError(
                "LBFGSB - one of the lower bounds is greater than an upper bound."
            )

        # initial vector must lie within the bounds. Otherwise ScalarFunction and
        # approx_derivative will cause problems
        x0 = np.clip(x0, bounds[0], bounds[1])

    # _prepare_scalar_function can use bounds=None to represent no bounds
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  bounds=bounds,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  workers=workers)

    func_and_grad = sf.fun_and_grad

    nbd = zeros(n, np.int32)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(-np.inf, np.inf): 0,
                  (1, np.inf): 1,
                  (1, 1): 2,
                  (-np.inf, 1): 3}

    if bounds is not None:
        for i in range(0, n):
            L, U = bounds[0, i], bounds[1, i]
            if not np.isinf(L):
                low_bnd[i] = L
                L = 1
            if not np.isinf(U):
                upper_bnd[i] = U
                U = 1
            nbd[i] = bounds_map[L, U]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    x = array(x0, dtype=np.float64)
    f = array(0.0, dtype=np.float64)
    g = zeros((n,), dtype=np.float64)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
    iwa = zeros(3*n, dtype=np.int32)
    task = zeros(2, dtype=np.int32)
    ln_task = zeros(2, dtype=np.int32)
    lsave = zeros(4, dtype=np.int32)
    isave = zeros(44, dtype=np.int32)
    dsave = zeros(29, dtype=float64)

    n_iterations = 0

    while True:
        # g may become float32 if a user provides a function that calculates
        # the Jacobian in float32 (see gh-18730). The underlying code expects
        # float64, so upcast it
        g = g.astype(np.float64)
        # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa,
                       iwa, task, lsave, isave, dsave, maxls, ln_task)

        if task[0] == 3:
            # The minimization routine wants f and g at the current x.
            # Note that interruptions due to maxfun are postponed
            # until the completion of the current minimization iteration.
            # Overwrite f and g:
            f, g = func_and_grad(x)
        elif task[0] == 1:
            # new iteration
            n_iterations += 1

            intermediate_result = OptimizeResult(x=x, fun=f)
            if _call_callback_maybe_halt(callback, intermediate_result):
                task[0] = 5
                task[1] = 505
            if n_iterations >= maxiter:
                task[0] = 5
                task[1] = 504
            elif sf.nfev > maxfun:
                task[0] = 5
                task[1] = 502
        else:
            break

    if task[0] == 4:
        warnflag = 0
    elif sf.nfev > maxfun or n_iterations >= maxiter:
        warnflag = 1
    else:
        warnflag = 2

    # These two portions of the workspace are described in the mainlb
    # function docstring in "__lbfgsb.c", ws and wy arguments.
    s = wa[0: m*n].reshape(m, n)
    y = wa[m*n: 2*m*n].reshape(m, n)

    # isave(31) = the total number of BFGS updates prior the current iteration.
    n_bfgs_updates = isave[30]

    n_corrs = min(n_bfgs_updates, maxcor)
    hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

    msg = status_messages[task[0]] + ": " + task_messages[task[1]]

    return OptimizeResult(fun=f, jac=g, nfev=sf.nfev,
                          njev=sf.ngev,
                          nit=n_iterations, status=warnflag, message=msg,
                          x=x, success=(warnflag == 0), hess_inv=hess_inv)


class LbfgsInvHessProduct(LinearOperator):
    """Linear operator for the L-BFGS approximate inverse Hessian.

    This operator computes the product of a vector with the approximate inverse
    of the Hessian of the objective function, using the L-BFGS limited
    memory approximation to the inverse Hessian, accumulated during the
    optimization.

    Objects of this class implement the ``scipy.sparse.linalg.LinearOperator``
    interface.

    Parameters
    ----------
    sk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the solution vector.
        (See [1]).
    yk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the gradient. (See [1]).

    References
    ----------
    .. [1] Nocedal, Jorge. "Updating quasi-Newton matrices with limited
       storage." Mathematics of computation 35.151 (1980): 773-782.

    """

    def __init__(self, sk, yk):
        """Construct the operator."""
        if sk.shape != yk.shape or sk.ndim != 2:
            raise ValueError('sk and yk must have matching shape, (n_corrs, n)')
        n_corrs, n = sk.shape

        super().__init__(dtype=np.float64, shape=(n, n))

        self.sk = sk
        self.yk = yk
        self.n_corrs = n_corrs
        self.rho = 1 / np.einsum('ij,ij->i', sk, yk)

    def _matvec(self, x):
        """Efficient matrix-vector multiply with the BFGS matrices.

        This calculation is described in Section (4) of [1].

        Parameters
        ----------
        x : ndarray
            An array with shape (n,) or (n,1).

        Returns
        -------
        y : ndarray
            The matrix-vector product

        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        q = np.array(x, dtype=self.dtype, copy=True)
        if q.ndim == 2 and q.shape[1] == 1:
            q = q.reshape(-1)

        alpha = np.empty(n_corrs)

        for i in range(n_corrs-1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q = q - alpha[i]*y[i]

        r = q
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)

        return r

    def _matmat(self, X):
        """Efficient matrix-matrix multiply with the BFGS matrices.

        This calculation is described in Section (4) of [1].

        Parameters
        ----------
        X : ndarray
            An array with shape (n,m)

        Returns
        -------
        Y : ndarray
            The matrix-matrix product

        Notes
        -----
        This implementation is written starting from _matvec and broadcasting
        all expressions along the second axis of X.

        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        Q = np.array(X, dtype=self.dtype, copy=True)

        alpha = np.empty((n_corrs, Q.shape[1]))

        for i in range(n_corrs-1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], Q)
            Q -= alpha[i]*y[i][:, np.newaxis]

        R = Q
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], R)
            R += s[i][:, np.newaxis] * (alpha[i] - beta)

        return R

    def todense(self):
        """Return a dense array representation of this operator.

        Returns
        -------
        arr : ndarray, shape=(n, n)
            An array with the same shape and containing
            the same data represented by this `LinearOperator`.

        """
        I_arr = np.eye(*self.shape, dtype=self.dtype)
        return self._matmat(I_arr)
