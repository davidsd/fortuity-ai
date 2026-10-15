/* Dense linear algebra mod p (p < 2^26) on top of BLIS dgemm, for hwcoh.
 *
 * Exactness: operands have entries in [0, p).  A product A (m x k) * B (k x n) is computed as
 *   A * Bhi * 2^13 + A * Blo,   B = Bhi * 2^13 + Blo,  0 <= Bhi, Blo < 2^13,
 * with k split into chunks of at most 2^14, so every partial sum is < 2^26 * 2^13 * 2^14 = 2^53 (exact in double).
 *
 * ff_matmul: C = A B mod p.
 * ff_rank2 : rank of the first `split` rows of A (r x c) and of all rows, by blocked forward elimination:
 *            the basis is a list of blocks, each in RREF w.r.t. its own pivots and zero on the pivots of earlier
 *            blocks; a new row block is reduced against the blocks in order (one gemm per block) and then
 *            eliminated internally (recursively, gemm-based).  Optionally reduces extra rows G against the final
 *            basis of the first `split` rows only (returns rank of [A_split; G] - rank(A_split) via ff_rank2).
 * Storage: int64 row-major (values < p); temporaries in double.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <blis/cblas.h>

#define KCHUNK 16384
static const double TWO13 = 8192.0;

static inline int64_t modp(int64_t x, int64_t p) { x %= p; return x < 0 ? x + p : x; }

static int64_t inv_mod(int64_t a, int64_t p) {
    int64_t r = 1, e = p - 2; a = modp(a, p);
    while (e) { if (e & 1) r = (__int128)r * a % p; a = (__int128)a * a % p; e >>= 1; }
    return r;
}

/* C (m x n, ldc) = A (m x k, lda, int64) * B (k x n, ldb) mod p; accumulate=1 adds to C (mod p).
   Generated for B, C element types (int64 / int32). */
#define DEFINE_MATMUL(NAME, BT, CT)                                                                          \
static void NAME(const int64_t *A, long lda, const BT *B, long ldb, CT *C, long ldc,                          \
                 long m, long k, long n, int64_t p, int accumulate) {                                        \
    long kc = k < KCHUNK ? k : KCHUNK;                                                                        \
    double *Ad = malloc(sizeof(double) * m * kc);                                                             \
    double *Bh = malloc(sizeof(double) * kc * n), *Bl = malloc(sizeof(double) * kc * n);                      \
    double *T1 = malloc(sizeof(double) * m * n), *T2 = malloc(sizeof(double) * m * n);                        \
    if (!accumulate) {                                                                                        \
        _Pragma("omp parallel for")                                                                           \
        for (long i = 0; i < m; i++) for (long j = 0; j < n; j++) C[i * ldc + j] = 0;                          \
    }                                                                                                         \
    double pd = (double)p;                                                                                    \
    for (long k0 = 0; k0 < k; k0 += KCHUNK) {                                                                 \
        long kk = (k - k0 < KCHUNK) ? k - k0 : KCHUNK;                                                        \
        _Pragma("omp parallel for")                                                                           \
        for (long i = 0; i < m; i++) for (long t = 0; t < kk; t++) Ad[i * kk + t] = (double)A[i * lda + k0 + t]; \
        _Pragma("omp parallel for")                                                                           \
        for (long t = 0; t < kk; t++) for (long j = 0; j < n; j++) {                                          \
            int64_t v = (int64_t)B[(k0 + t) * ldb + j];                                                       \
            Bh[t * n + j] = (double)(v >> 13); Bl[t * n + j] = (double)(v & 8191); }                           \
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, kk, 1.0, Ad, kk, Bh, n, 0.0, T1, n);     \
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, kk, 1.0, Ad, kk, Bl, n, 0.0, T2, n);     \
        _Pragma("omp parallel for")                                                                           \
        for (long i = 0; i < m; i++) for (long j = 0; j < n; j++) {                                           \
            double h = fmod(T1[i * n + j], pd), l = fmod(T2[i * n + j], pd);                                  \
            int64_t v = (int64_t)C[i * ldc + j] + (int64_t)fmod(h * TWO13 + l, pd);                           \
            C[i * ldc + j] = (CT)(v >= p ? v - p : v); }                                                      \
    }                                                                                                         \
    free(Ad); free(Bh); free(Bl); free(T1); free(T2);                                                         \
}
DEFINE_MATMUL(matmul_core, int64_t, int64_t)
DEFINE_MATMUL(matmul_core_b32, int32_t, int64_t)
DEFINE_MATMUL(matmul_core_b32c32, int32_t, int32_t)

void ff_set_threads(int t) { omp_set_num_threads(t); }

void ff_matmul(const int64_t *A, long m, long k, const int64_t *B, long n, int64_t *C, int64_t p) {
    if (m == 0 || n == 0) return;
    if (k == 0) { memset(C, 0, sizeof(int64_t) * m * n); return; }
    matmul_core(A, k, B, n, C, n, m, k, n, p, 0);
}

/* ---------------- blocked elimination ---------------- */

typedef struct { long nrows; long *piv; int32_t *E; } Block;     /* E: nrows x c, RREF on piv (int32) */
typedef struct { long c; int64_t p; long nblocks, cap, rank; Block *b; } Basis;

static void basis_init(Basis *B, long c, int64_t p) { B->c = c; B->p = p; B->nblocks = 0; B->cap = 16; B->rank = 0;
                                                     B->b = malloc(sizeof(Block) * B->cap); }
static void basis_free(Basis *B) { for (long i = 0; i < B->nblocks; i++) { free(B->b[i].piv); free(B->b[i].E); } free(B->b); }

/* R (r x c) -= R[:, piv_k] * E_k for all blocks, in order */
static void reduce_against(const Basis *B, int64_t *R, long r) {
    long c = B->c; int64_t p = B->p;
    for (long k = 0; k < B->nblocks; k++) {
        const Block *bk = &B->b[k];
        long nb = bk->nrows;
        int64_t *coef = malloc(sizeof(int64_t) * r * nb);
        #pragma omp parallel for
        for (long i = 0; i < r; i++)
            for (long t = 0; t < nb; t++) { int64_t v = R[i * c + bk->piv[t]]; coef[i * nb + t] = v ? p - v : 0; }
        matmul_core_b32(coef, nb, bk->E, c, R, c, r, nb, c, p, 1);   /* R += (-coef) E */
        free(coef);
    }
}

/* In-place RREF of R (r x c) restricted to rows; returns rank rr, pivots in piv (length rr); rows 0..rr-1 hold the
   RREF rows (others garbage).  Recursive: eliminate top half, reduce bottom half, eliminate bottom half, back-reduce. */
static long rref_block(int64_t *R, long r, long c, int64_t p, long *piv) {
    if (r <= 32) {
        long rk = 0;
        for (long col = 0; col < c && rk < r; col++) {
            long sel = -1;
            for (long i = rk; i < r; i++) if (R[i * c + col]) { sel = i; break; }
            if (sel < 0) continue;
            if (sel != rk) for (long j = 0; j < c; j++) { int64_t t = R[rk * c + j]; R[rk * c + j] = R[sel * c + j]; R[sel * c + j] = t; }
            int64_t iv = inv_mod(R[rk * c + col], p);
            #pragma omp parallel for
            for (long j = 0; j < c; j++) R[rk * c + j] = (int64_t)((__int128)R[rk * c + j] * iv % p);
            for (long i = 0; i < r; i++) {
                if (i == rk) continue;
                int64_t f = R[i * c + col];
                if (!f) continue;
                #pragma omp parallel for
                for (long j = 0; j < c; j++) R[i * c + j] = modp(R[i * c + j] - (int64_t)((__int128)f * R[rk * c + j] % p), p);
            }
            piv[rk++] = col;
        }
        return rk;
    }
    long h = r / 2;
    long *pt = malloc(sizeof(long) * h), *pb = malloc(sizeof(long) * (r - h));
    long rt = rref_block(R, h, c, p, pt);
    int64_t *Bt = R + h * c; long rb_rows = r - h;
    if (rt) {
        int64_t *coef = malloc(sizeof(int64_t) * rb_rows * rt);
        #pragma omp parallel for
        for (long i = 0; i < rb_rows; i++)
            for (long t = 0; t < rt; t++) { int64_t v = Bt[i * c + pt[t]]; coef[i * rt + t] = v ? p - v : 0; }
        matmul_core(coef, rt, R, c, Bt, c, rb_rows, rt, c, p, 1);
        free(coef);
    }
    long rb = rref_block(Bt, rb_rows, c, p, pb);
    if (rb && rt) {                                      /* back-reduce top rows on the bottom pivots */
        int64_t *coef = malloc(sizeof(int64_t) * rt * rb);
        #pragma omp parallel for
        for (long i = 0; i < rt; i++)
            for (long t = 0; t < rb; t++) { int64_t v = R[i * c + pb[t]]; coef[i * rb + t] = v ? p - v : 0; }
        matmul_core(coef, rb, Bt, c, R, c, rt, rb, c, p, 1);
        free(coef);
    }
    /* compact: bottom RREF rows right after the top ones */
    if (rb && rt != h) memmove(R + rt * c, Bt, sizeof(int64_t) * rb * c);
    for (long t = 0; t < rt; t++) piv[t] = pt[t];
    for (long t = 0; t < rb; t++) piv[rt + t] = pb[t];
    free(pt); free(pb);
    return rt + rb;
}

/* add rows (r x c) to the basis; returns number of new independent rows.  Rows are consumed (modified). */
static long basis_add(Basis *B, int64_t *R, long r) {
    if (r == 0) return 0;
    reduce_against(B, R, r);
    long *piv = malloc(sizeof(long) * (r < B->c ? r : B->c));
    long rk = rref_block(R, r, B->c, B->p, piv);
    if (rk == 0) { free(piv); return 0; }
    if (B->nblocks == B->cap) { B->cap *= 2; B->b = realloc(B->b, sizeof(Block) * B->cap); }
    Block *nb = &B->b[B->nblocks++];
    nb->nrows = rk; nb->piv = realloc(piv, sizeof(long) * rk);
    nb->E = malloc(sizeof(int32_t) * rk * B->c);
    #pragma omp parallel for
    for (long i = 0; i < rk * B->c; i++) nb->E[i] = (int32_t)R[i];
    B->rank += rk;
    return rk;
}

/* rank of A[0:split] (-> *rank_split) and of A (return value).  A is not modified.  blk: row block size. */
long ff_rank2(const int64_t *A, long r, long c, long split, int64_t p, long blk, long *rank_split) {
    Basis B; basis_init(&B, c, p);
    if (blk <= 0) blk = 1024;
    int64_t *R = malloc(sizeof(int64_t) * blk * c);
    long rs = -1;
    for (long r0 = 0; r0 < r; ) {
        long end = r0 + blk;
        if (r0 < split && end > split) end = split;
        if (end > r) end = r;
        memcpy(R, A + r0 * c, sizeof(int64_t) * (end - r0) * c);
        basis_add(&B, R, end - r0);
        r0 = end;
        if (r0 == split) rs = B.rank;
    }
    if (rs < 0) rs = (split <= 0) ? 0 : B.rank;
    long tot = B.rank;
    free(R); basis_free(&B);
    if (rank_split) *rank_split = rs;
    return tot;
}

/* C[:, 0:n] (int32, ldc) = A (m x k int64) * B (k x n int32, ldb) mod p  (overwrite) */
void ff_matmul_b32c32(const int64_t *A, long m, long k, const int32_t *B, long ldb, long n, int32_t *C, long ldc,
                      int64_t p) {
    if (m == 0 || n == 0) return;
    if (k == 0) { for (long i = 0; i < m; i++) memset(C + i * ldc, 0, sizeof(int32_t) * n); return; }
    matmul_core_b32c32(A, k, B, ldb, C, ldc, m, k, n, p, 0);
}

/* rank of A1 (r1 x c, int32, row stride lda1) -> *rank1, and of [A1; A2] (return).  Row blocks of blk rows. */
long ff_rank_pair_i32(const int32_t *A1, long lda1, long r1, const int32_t *A2, long lda2, long r2, long c,
                      int64_t p, long blk, long *rank1) {
    Basis B; basis_init(&B, c, p);
    if (blk <= 0) blk = 4096;
    int64_t *R = malloc(sizeof(int64_t) * blk * c);
    for (int part = 0; part < 2; part++) {
        const int32_t *A = part ? A2 : A1; long lda = part ? lda2 : lda1; long r = part ? r2 : r1;
        for (long r0 = 0; r0 < r; r0 += blk) {
            long nr = (r - r0 < blk) ? r - r0 : blk;
            #pragma omp parallel for
            for (long i = 0; i < nr; i++) for (long j = 0; j < c; j++) R[i * c + j] = A[(r0 + i) * lda + j];
            basis_add(&B, R, nr);
        }
        if (part == 0 && rank1) *rank1 = B.rank;
    }
    long tot = B.rank;
    free(R); basis_free(&B);
    return tot;
}

/* Right null space of A (r x c, int64, entries in [0,p)): basis rows written to out (nullity x c, caller allocates
   c x c), returns the nullity.  Blocked RREF (basis_add), then back-reduction across blocks (later pivots eliminated
   from earlier blocks: full RREF), then x_f = e_f - sum_t R[t, f] e_{piv_t} for each free column f. */
long ff_nullspace(const int64_t *A, long r, long c, int64_t p, long blk, int64_t *out) {
    Basis B; basis_init(&B, c, p);
    if (blk <= 0) blk = 4096;
    int64_t *R = malloc(sizeof(int64_t) * blk * c);
    for (long r0 = 0; r0 < r; r0 += blk) {
        long nr = (r - r0 < blk) ? r - r0 : blk;
        memcpy(R, A + r0 * c, sizeof(int64_t) * nr * c);
        basis_add(&B, R, nr);
    }
    free(R);
    /* back-reduction: for k from last to first, eliminate piv_k from all earlier blocks */
    for (long k = B.nblocks - 1; k >= 1; k--) {
        Block *bk = &B.b[k];
        for (long j = 0; j < k; j++) {
            Block *bj = &B.b[j];
            int64_t *coef = malloc(sizeof(int64_t) * bj->nrows * bk->nrows);
            #pragma omp parallel for
            for (long i = 0; i < bj->nrows; i++)
                for (long t = 0; t < bk->nrows; t++) { int64_t v = bj->E[i * c + bk->piv[t]]; coef[i * bk->nrows + t] = v ? p - v : 0; }
            matmul_core_b32c32(coef, bk->nrows, bk->E, c, bj->E, c, bj->nrows, bk->nrows, c, p, 1);
            free(coef);
        }
    }
    char *ispiv = calloc(c, 1);
    for (long k = 0; k < B.nblocks; k++) for (long t = 0; t < B.b[k].nrows; t++) ispiv[B.b[k].piv[t]] = 1;
    long *fidx = malloc(sizeof(long) * c);          /* column -> free index or -1 */
    long nul = 0;
    for (long j = 0; j < c; j++) fidx[j] = ispiv[j] ? -1 : nul++;
    memset(out, 0, sizeof(int64_t) * nul * c);
    for (long j = 0; j < c; j++) if (fidx[j] >= 0) out[fidx[j] * c + j] = 1;
    for (long k = 0; k < B.nblocks; k++) {
        Block *bk = &B.b[k];
        #pragma omp parallel for
        for (long t = 0; t < bk->nrows; t++) {
            long pc = bk->piv[t];
            for (long j = 0; j < c; j++) if (fidx[j] >= 0) {
                int64_t v = bk->E[t * c + j];
                out[fidx[j] * c + pc] = v ? p - v : 0;
            }
        }
    }
    free(ispiv); free(fidx); basis_free(&B);
    return nul;
}

/* Incremental basis API (streaming rows): new / add row blocks / null space of the span / free. */
void *ff_basis_new(long c, int64_t p) { Basis *B = malloc(sizeof(Basis)); basis_init(B, c, p); return B; }
long ff_basis_rank(void *h) { return ((Basis *)h)->rank; }
long ff_basis_add(void *h, const int64_t *A, long r) {
    Basis *B = (Basis *)h;
    int64_t *R = malloc(sizeof(int64_t) * (r ? r : 1) * B->c);
    memcpy(R, A, sizeof(int64_t) * r * B->c);
    long k = basis_add(B, R, r);
    free(R);
    return k;
}
void ff_basis_free(void *h) { basis_free((Basis *)h); free(h); }

/* Null space {x : v.x = 0 for all rows v of the basis}: writes rows to out (maxnul x c); returns the nullity, or
   -(nullity) without writing when maxnul is too small.  Performs the back-reduction (destructive). */
long ff_basis_nullspace(void *h, int64_t *out, long maxnul) {
    Basis *Bp = (Basis *)h; long c = Bp->c; int64_t p = Bp->p;
    long nul = c - Bp->rank;
    if (nul > maxnul) return -nul;
    for (long k = Bp->nblocks - 1; k >= 1; k--) {
        Block *bk = &Bp->b[k];
        for (long j = 0; j < k; j++) {
            Block *bj = &Bp->b[j];
            int64_t *coef = malloc(sizeof(int64_t) * bj->nrows * bk->nrows);
            #pragma omp parallel for
            for (long i = 0; i < bj->nrows; i++)
                for (long t = 0; t < bk->nrows; t++) { int64_t v = bj->E[i * c + bk->piv[t]]; coef[i * bk->nrows + t] = v ? p - v : 0; }
            matmul_core_b32c32(coef, bk->nrows, bk->E, c, bj->E, c, bj->nrows, bk->nrows, c, p, 1);
            free(coef);
        }
    }
    char *ispiv = calloc(c, 1);
    for (long k = 0; k < Bp->nblocks; k++) for (long t = 0; t < Bp->b[k].nrows; t++) ispiv[Bp->b[k].piv[t]] = 1;
    long *fidx = malloc(sizeof(long) * c); long f = 0;
    for (long j = 0; j < c; j++) fidx[j] = ispiv[j] ? -1 : f++;
    memset(out, 0, sizeof(int64_t) * nul * c);
    for (long j = 0; j < c; j++) if (fidx[j] >= 0) out[fidx[j] * c + j] = 1;
    for (long k = 0; k < Bp->nblocks; k++) {
        Block *bk = &Bp->b[k];
        #pragma omp parallel for
        for (long t = 0; t < bk->nrows; t++) {
            long pc = bk->piv[t];
            for (long j = 0; j < c; j++) if (fidx[j] >= 0) { int64_t v = bk->E[t * c + j]; out[fidx[j] * c + pc] = v ? p - v : 0; }
        }
    }
    free(ispiv); free(fidx);
    return nul;
}
