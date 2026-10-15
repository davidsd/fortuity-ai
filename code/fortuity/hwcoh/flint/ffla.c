/* Thin FLINT wrapper for dense linear algebra mod p on int64 row-major arrays (entries in [0, p)).
   Built against FLINT 3.0.1 (installation prefix ${FLINT_PREFIX}); called from Python via ctypes (hwcoh/ffla.py). */
#include <string.h>
#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <flint/thread_support.h>

static void load(nmod_mat_t M, const long *A, long r, long c) {
    for (long i = 0; i < r; i++) memcpy(M->rows[i], A + i * c, c * sizeof(ulong));
}

void ff_set_threads(int t) { flint_set_num_threads(t); }

long ff_rank(const long *A, long r, long c, unsigned long p) {
    nmod_mat_t M; nmod_mat_init(M, r, c, p); load(M, A, r, c);
    long rk = nmod_mat_rank(M); nmod_mat_clear(M); return rk;
}

/* right null space of A (r x c): writes up to c basis vectors as rows of out (c x c buffer), returns nullity */
long ff_nullspace(const long *A, long r, long c, unsigned long p, long *out) {
    nmod_mat_t M, X; nmod_mat_init(M, r, c, p); load(M, A, r, c);
    nmod_mat_init(X, c, c, p);
    long nul = nmod_mat_nullspace(X, M);
    for (long j = 0; j < nul; j++) for (long i = 0; i < c; i++) out[j * c + i] = (long) nmod_mat_entry(X, i, j);
    nmod_mat_clear(M); nmod_mat_clear(X); return nul;
}

/* in-place RREF of A (r x c); returns rank; pivots written to piv (length >= rank) */
long ff_rref(long *A, long r, long c, unsigned long p, long *piv) {
    nmod_mat_t M; nmod_mat_init(M, r, c, p); load(M, A, r, c);
    long rk = nmod_mat_rref(M);
    for (long i = 0; i < r; i++) memcpy(A + i * c, M->rows[i], c * sizeof(ulong));
    long k = 0;
    for (long i = 0; i < rk; i++) { long j = 0; while (j < c && nmod_mat_entry(M, i, j) == 0) j++; piv[k++] = j; }
    nmod_mat_clear(M); return rk;
}
