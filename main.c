#define _GNU_SOURCE
// High-Concurrency MIMO VRFT HTTP/1.1 Server with AVX2 + FMA Accelerated Engine
// Supports K8s /healthz, Persistent HTTP Keep-Alive, and Built-in AVX2 Speedup Benchmark

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define MAX_QUEUE 1024
#define WORKER_THREADS 8

typedef struct {
    int num_len;
    int den_len;
    double *num;
    double *den;
} DiscreteTF;

typedef struct {
    int rows;
    int cols;
    DiscreteTF *elements;
} MimoTF;

typedef struct {
    int n_samples;
    int n_channels;
    int total_params;
    double *u;
    double *y;
    double *y_iv;
    MimoTF Td;
    MimoTF L;
    int *c_len;
    DiscreteTF **C_basis;
    double *p_lower;
    double *p_upper;
} VrftRequest;

typedef int (*InversionStrategy)(const MimoTF *Td, const double *y_in, int N, int n, double *rv_out, int *out_N);
typedef void (*OptimizerStrategy)(const double *Z, const double *Y, const double *lb, const double *ub, int P, double *p_out);

typedef struct {
    InversionStrategy invert_model;
    OptimizerStrategy solve_constrained;
} VrftPipeline;

typedef struct {
    int client_fd;
    const VrftPipeline *pipeline;
} JobItem;

typedef struct {
    JobItem buffer[MAX_QUEUE];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t notify;
} ThreadPoolQueue;

static inline double hsum256_pd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

static inline double avx2_dot_product(const double *a, const double *b, int len) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int k = 0;
    for (; k <= len - 8; k += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k), _mm256_loadu_pd(b + k), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k + 4), _mm256_loadu_pd(b + k + 4), acc1);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    for (; k <= len - 4; k += 4) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + k), _mm256_loadu_pd(b + k), acc0);
    }
    double sum = hsum256_pd(acc0);
    for (; k < len; ++k) sum += a[k] * b[k];
    return sum;
}

static void dlsim_siso(const DiscreteTF *tf, const double *u, int N, double *y_out) {
    if (tf->num_len == 0 || tf->den_len == 0 || fabs(tf->den[0]) < 1e-15) return;
    int max_ord = (tf->num_len > tf->den_len ? tf->num_len : tf->den_len);
    double *state = calloc(max_ord, sizeof(double));
    double a0_inv = 1.0 / tf->den[0];
    double b0 = tf->num[0] * a0_inv;

    for (int k = 0; k < N; ++k) {
        double u_k = u[k];
        double y_k = b0 * u_k + state[0];
        y_out[k] += y_k;
        for (int i = 1; i < max_ord; ++i) {
            double b_i = (i < tf->num_len) ? tf->num[i] * a0_inv : 0.0;
            double a_i = (i < tf->den_len) ? tf->den[i] * a0_inv : 0.0;
            state[i - 1] = b_i * u_k - a_i * y_k + state[i];
        }
    }
    free(state);
}

static void dlsim_siso_2x_avx(const DiscreteTF *tf, const double *u0, const double *u1, int N, double *y0, double *y1) {
    if (tf->num_len == 0 || tf->den_len == 0 || fabs(tf->den[0]) < 1e-15) return;
    int max_ord = (tf->num_len > tf->den_len ? tf->num_len : tf->den_len);
    __m128d *state = aligned_alloc(16, sizeof(__m128d) * max_ord);
    for (int i = 0; i < max_ord; ++i) state[i] = _mm_setzero_pd();

    double a0_inv = 1.0 / tf->den[0];
    __m128d v_b0 = _mm_set1_pd(tf->num[0] * a0_inv);

    for (int k = 0; k < N; ++k) {
        __m128d v_u = _mm_set_pd(u1[k], u0[k]);
        __m128d v_y = _mm_fmadd_pd(v_b0, v_u, state[0]);
        y0[k] += _mm_cvtsd_f64(v_y);
        y1[k] += _mm_cvtsd_f64(_mm_unpackhi_pd(v_y, v_y));

        for (int i = 1; i < max_ord; ++i) {
            __m128d v_bi = _mm_set1_pd((i < tf->num_len) ? tf->num[i] * a0_inv : 0.0);
            __m128d v_ai = _mm_set1_pd((i < tf->den_len) ? tf->den[i] * a0_inv : 0.0);
            state[i - 1] = _mm_fmadd_pd(v_bi, v_u, _mm_fnmadd_pd(v_ai, v_y, state[i]));
        }
    }
    free(state);
}

static void col_filter_avx2(const DiscreteTF *G, const double *u_mat, int N, int cols, double *y_out) {
    memset(y_out, 0, sizeof(double) * N * cols);
    if (G->num_len == 0 || G->den_len == 0 || fabs(G->den[0]) < 1e-15) return;

    int max_ord = (G->num_len > G->den_len ? G->num_len : G->den_len);
    double a0_inv = 1.0 / G->den[0];
    __m256d v_b0 = _mm256_set1_pd(G->num[0] * a0_inv);

    int c = 0;
    for (; c <= cols - 4; c += 4) {
        __m256d *state = aligned_alloc(32, sizeof(__m256d) * max_ord);
        for (int i = 0; i < max_ord; ++i) state[i] = _mm256_setzero_pd();

        const double *u0 = u_mat + (c + 0) * N, *u1 = u_mat + (c + 1) * N;
        const double *u2 = u_mat + (c + 2) * N, *u3 = u_mat + (c + 3) * N;
        double *y0 = y_out + (c + 0) * N, *y1 = y_out + (c + 1) * N;
        double *y2 = y_out + (c + 2) * N, *y3 = y_out + (c + 3) * N;

        for (int k = 0; k < N; ++k) {
            __m256d v_u = _mm256_set_pd(u3[k], u2[k], u1[k], u0[k]);
            __m256d v_y = _mm256_fmadd_pd(v_b0, v_u, state[0]);
            double tmp[4];
            _mm256_storeu_pd(tmp, v_y);
            y0[k] = tmp[0]; y1[k] = tmp[1]; y2[k] = tmp[2]; y3[k] = tmp[3];

            for (int i = 1; i < max_ord; ++i) {
                __m256d v_bi = _mm256_set1_pd((i < G->num_len) ? G->num[i] * a0_inv : 0.0);
                __m256d v_ai = _mm256_set1_pd((i < G->den_len) ? G->den[i] * a0_inv : 0.0);
                state[i - 1] = _mm256_fmadd_pd(v_bi, v_u, _mm256_fnmadd_pd(v_ai, v_y, state[i]));
            }
        }
        free(state);
    }
    for (; c <= cols - 2; c += 2) {
        dlsim_siso_2x_avx(G, u_mat + c * N, u_mat + (c + 1) * N, N, y_out + c * N, y_out + (c + 1) * N);
    }
    for (; c < cols; ++c) {
        dlsim_siso(G, u_mat + c * N, N, y_out + c * N);
    }
}

static void mimo_filter(const MimoTF *G, const double *u, int N, double *y_out) {
    memset(y_out, 0, sizeof(double) * N * G->rows);
    for (int i = 0; i < G->rows; ++i) {
        for (int j = 0; j < G->cols; ++j) {
            const DiscreteTF *tf = &G->elements[i * G->cols + j];
            dlsim_siso(tf, u + j * N, N, y_out + i * N);
        }
    }
}

static int default_stable_inversion(const MimoTF *Td, const double *y_in, int N, int n, double *rv_out, int *out_N) {
    int max_delay = 1;
    memset(rv_out, 0, sizeof(double) * N * n);

    for (int i = 0; i < n; ++i) {
        const DiscreteTF *diag_tf = &Td->elements[i * n + i];
        if (diag_tf->num_len == 0) return 1;
        int delay = 0;
        while (delay < diag_tf->num_len && fabs(diag_tf->num[delay]) < 1e-12) delay++;
        if (delay >= diag_tf->num_len) return 1;
        if (delay > max_delay) max_delay = delay;

        DiscreteTF inv_tf = {
            .num_len = diag_tf->den_len,
            .den_len = diag_tf->num_len - delay,
            .num = diag_tf->den,
            .den = diag_tf->num + delay
        };
        if (inv_tf.den_len > 1 && fabs(inv_tf.den[1] / inv_tf.den[0]) >= 1.0) return 2;
        dlsim_siso(&inv_tf, y_in + i * N + delay, N - delay, rv_out + i * N);
    }
    *out_N = N - max_delay;
    return (*out_N > 0) ? 0 : 1;
}

static void solve_box_qp_cd_avx2(const double *Z, const double *Y, const double *lb, const double *ub, int P, double *p_out) {
    double *ZT = calloc(P * P, sizeof(double));
    double *Q = calloc(P * P, sizeof(double));
    double *c = calloc(P, sizeof(double));

    for (int i = 0; i < P; ++i) {
        for (int k = 0; k < P; ++k) ZT[i * P + k] = Z[k * P + i];
    }

    for (int i = 0; i < P; ++i) {
        const double *zt_i = ZT + i * P;
        for (int j = i; j < P; ++j) {
            double sum = avx2_dot_product(zt_i, ZT + j * P, P);
            if (i == j) sum += 1e-9;
            Q[i * P + j] = sum;
            Q[j * P + i] = sum;
        }
        c[i] = avx2_dot_product(zt_i, Y, P);
        p_out[i] = fmin(ub[i], fmax(lb[i], 0.0));
    }

    for (int iter = 0; iter < 1500; ++iter) {
        double max_diff = 0.0;
        for (int i = 0; i < P; ++i) {
            const double *q_row = Q + i * P;
            double q_ii = q_row[i];
            double dot_all = avx2_dot_product(q_row, p_out, P);
            double r_i = c[i] - (dot_all - q_ii * p_out[i]);
            double next_pi = fmin(ub[i], fmax(lb[i], r_i / q_ii));
            max_diff = fmax(max_diff, fabs(next_pi - p_out[i]));
            p_out[i] = next_pi;
        }
        if (max_diff < 1e-10) break;
    }
    free(ZT); free(Q); free(c);
}

static int vrft_design_avx2(const VrftRequest *req, const VrftPipeline *pipe, double *p_out) {
    int N = req->n_samples, n = req->n_channels, P = req->total_params, N_valid = 0;
    double *uf = calloc(N * n, sizeof(double));
    double *rv = calloc(N * n, sizeof(double));
    double *rv_iv = calloc(N * n, sizeof(double));

    mimo_filter(&req->L, req->u, N, uf);
    int flag_vr = pipe->invert_model(&req->Td, req->y, N, n, rv, &N_valid);
    int dummy_N = 0;
    pipe->invert_model(&req->Td, req->y_iv, N, n, rv_iv, &dummy_N);

    if (flag_vr != 0) {
        free(uf); free(rv); free(rv_iv);
        return flag_vr;
    }

    double *ebar = calloc(N_valid * n, sizeof(double));
    double *ebar_iv = calloc(N_valid * n, sizeof(double));
    for (int ch = 0; ch < n; ++ch) {
        const double *rv_ch = rv + ch * N, *y_ch = req->y + ch * N;
        const double *rv_iv_ch = rv_iv + ch * N, *y_iv_ch = req->y_iv + ch * N;
        double *eb_ch = ebar + ch * N_valid, *eb_iv_ch = ebar_iv + ch * N_valid;
        int k = 0;
        for (; k <= N_valid - 4; k += 4) {
            _mm256_storeu_pd(eb_ch + k, _mm256_sub_pd(_mm256_loadu_pd(rv_ch + k), _mm256_loadu_pd(y_ch + k)));
            _mm256_storeu_pd(eb_iv_ch + k, _mm256_sub_pd(_mm256_loadu_pd(rv_iv_ch + k), _mm256_loadu_pd(y_iv_ch + k)));
        }
        for (; k < N_valid; ++k) {
            eb_ch[k] = rv_ch[k] - y_ch[k];
            eb_iv_ch[k] = rv_iv_ch[k] - y_iv_ch[k];
        }
    }

    int *row_param_counts = calloc(n, sizeof(int));
    int *row_param_offsets = calloc(n, sizeof(int));
    for (int i = 0; i < n; ++i) {
        row_param_offsets[i] = (i == 0) ? 0 : row_param_offsets[i - 1] + row_param_counts[i - 1];
        for (int j = 0; j < n; ++j) row_param_counts[i] += req->c_len[i * n + j];
    }

    double **phi_list = calloc(n, sizeof(double *));
    double **csi_list = calloc(n, sizeof(double *));
    for (int i = 0; i < n; ++i) {
        phi_list[i] = calloc(N_valid * row_param_counts[i], sizeof(double));
        csi_list[i] = calloc(N_valid * row_param_counts[i], sizeof(double));
        int col_cursor = 0;
        for (int j = 0; j < n; ++j) {
            int sub_len = req->c_len[i * n + j];
            for (int k = 0; k < sub_len; ++k) {
                dlsim_siso_2x_avx(&req->C_basis[i * n + j][k],
                                  ebar + j * N_valid, ebar_iv + j * N_valid, N_valid,
                                  phi_list[i] + col_cursor * N_valid, csi_list[i] + col_cursor * N_valid);
                col_cursor++;
            }
        }
    }

    int total_rows = n * N_valid;
    // Column-Major storage: column `c` is contiguous at `Phi_col + c * total_rows`
    double *Phi_col = calloc(total_rows * P, sizeof(double));
    double *Csi_col = calloc(total_rows * P, sizeof(double));
    double *Uf_vec = calloc(total_rows, sizeof(double));

    for (int i = 0; i < n; ++i) {
        memcpy(Uf_vec + i * N_valid, uf + i * N, sizeof(double) * N_valid);
        for (int j = 0; j < n; ++j) {
            int p_cnt = row_param_counts[j];
            if (p_cnt == 0) continue;
            double *tmp_phi = calloc(N_valid * p_cnt, sizeof(double));
            double *tmp_csi = calloc(N_valid * p_cnt, sizeof(double));
            col_filter_avx2(&req->L.elements[i * n + j], phi_list[j], N_valid, p_cnt, tmp_phi);
            col_filter_avx2(&req->L.elements[i * n + j], csi_list[j], N_valid, p_cnt, tmp_csi);

            for (int c = 0; c < p_cnt; ++c) {
                int target_col = row_param_offsets[j] + c;
                memcpy(Phi_col + target_col * total_rows + i * N_valid, tmp_phi + c * N_valid, sizeof(double) * N_valid);
                memcpy(Csi_col + target_col * total_rows + i * N_valid, tmp_csi + c * N_valid, sizeof(double) * N_valid);
            }
            free(tmp_phi); free(tmp_csi);
        }
    }

    double *Z = calloc(P * P, sizeof(double));
    double *Y = calloc(P, sizeof(double));
    for (int i = 0; i < P; ++i) {
        const double *csi_i = Csi_col + i * total_rows;
        Y[i] = avx2_dot_product(csi_i, Uf_vec, total_rows);
        int j = 0;
        for (; j <= P - 2; j += 2) {
            const double *phi_j0 = Phi_col + j * total_rows;
            const double *phi_j1 = Phi_col + (j + 1) * total_rows;
            __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
            int r = 0;
            for (; r <= total_rows - 4; r += 4) {
                __m256d v_csi = _mm256_loadu_pd(csi_i + r);
                acc0 = _mm256_fmadd_pd(v_csi, _mm256_loadu_pd(phi_j0 + r), acc0);
                acc1 = _mm256_fmadd_pd(v_csi, _mm256_loadu_pd(phi_j1 + r), acc1);
            }
            double s0 = hsum256_pd(acc0), s1 = hsum256_pd(acc1);
            for (; r < total_rows; ++r) {
                s0 += csi_i[r] * phi_j0[r];
                s1 += csi_i[r] * phi_j1[r];
            }
            Z[i * P + j] = s0;
            Z[i * P + j + 1] = s1;
        }
        for (; j < P; ++j) {
            Z[i * P + j] = avx2_dot_product(csi_i, Phi_col + j * total_rows, total_rows);
        }
    }

    pipe->solve_constrained(Z, Y, req->p_lower, req->p_upper, P, p_out);

    for (int i = 0; i < n; ++i) { free(phi_list[i]); free(csi_list[i]); }
    free(phi_list); free(csi_list);
    free(row_param_counts); free(row_param_offsets);
    free(uf); free(rv); free(rv_iv); free(ebar); free(ebar_iv);
    free(Phi_col); free(Csi_col); free(Uf_vec); free(Z); free(Y);
    return 0;
}

// Legacy Scalar Engine kept exclusively for `--benchmark` speedup verification
static void col_filter_scalar(const DiscreteTF *G, const double *u_mat, int N, int cols, double *y_out) {
    memset(y_out, 0, sizeof(double) * N * cols);
    for (int c = 0; c < cols; ++c) dlsim_siso(G, u_mat + c * N, N, y_out + c * N);
}

static void solve_box_qp_cd_scalar(const double *Z, const double *Y, const double *lb, const double *ub, int P, double *p_out) {
    double *Q = calloc(P * P, sizeof(double)), *c = calloc(P, sizeof(double));
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = (i == j) ? 1e-9 : 0.0;
            for (int k = 0; k < P; ++k) sum += Z[k * P + i] * Z[k * P + j];
            Q[i * P + j] = sum;
        }
        double c_sum = 0.0;
        for (int k = 0; k < P; ++k) c_sum += Z[k * P + i] * Y[k];
        c[i] = c_sum;
        p_out[i] = fmin(ub[i], fmax(lb[i], 0.0));
    }
    for (int iter = 0; iter < 1500; ++iter) {
        double max_diff = 0.0;
        for (int i = 0; i < P; ++i) {
            double r_i = c[i];
            for (int j = 0; j < P; ++j) if (i != j) r_i -= Q[i * P + j] * p_out[j];
            double next_pi = fmin(ub[i], fmax(lb[i], r_i / Q[i * P + i]));
            max_diff = fmax(max_diff, fabs(next_pi - p_out[i]));
            p_out[i] = next_pi;
        }
        if (max_diff < 1e-10) break;
    }
    free(Q); free(c);
}

static int vrft_design_scalar(const VrftRequest *req, const VrftPipeline *pipe, double *p_out) {
    int N = req->n_samples, n = req->n_channels, P = req->total_params, N_valid = 0;
    double *uf = calloc(N * n, sizeof(double)), *rv = calloc(N * n, sizeof(double)), *rv_iv = calloc(N * n, sizeof(double));
    mimo_filter(&req->L, req->u, N, uf);
    int flag_vr = pipe->invert_model(&req->Td, req->y, N, n, rv, &N_valid), dummy_N = 0;
    pipe->invert_model(&req->Td, req->y_iv, N, n, rv_iv, &dummy_N);
    if (flag_vr != 0) { free(uf); free(rv); free(rv_iv); return flag_vr; }

    double *ebar = calloc(N_valid * n, sizeof(double)), *ebar_iv = calloc(N_valid * n, sizeof(double));
    for (int ch = 0; ch < n; ++ch) {
        for (int k = 0; k < N_valid; ++k) {
            ebar[ch * N_valid + k] = rv[ch * N + k] - req->y[ch * N + k];
            ebar_iv[ch * N_valid + k] = rv_iv[ch * N + k] - req->y_iv[ch * N + k];
        }
    }
    int *row_param_counts = calloc(n, sizeof(int)), *row_param_offsets = calloc(n, sizeof(int));
    for (int i = 0; i < n; ++i) {
        row_param_offsets[i] = (i == 0) ? 0 : row_param_offsets[i - 1] + row_param_counts[i - 1];
        for (int j = 0; j < n; ++j) row_param_counts[i] += req->c_len[i * n + j];
    }
    double **phi_list = calloc(n, sizeof(double *)), **csi_list = calloc(n, sizeof(double *));
    for (int i = 0; i < n; ++i) {
        phi_list[i] = calloc(N_valid * row_param_counts[i], sizeof(double));
        csi_list[i] = calloc(N_valid * row_param_counts[i], sizeof(double));
        int col_cursor = 0;
        for (int j = 0; j < n; ++j) {
            int sub_len = req->c_len[i * n + j];
            for (int k = 0; k < sub_len; ++k) {
                dlsim_siso(&req->C_basis[i * n + j][k], ebar + j * N_valid, N_valid, phi_list[i] + col_cursor * N_valid);
                dlsim_siso(&req->C_basis[i * n + j][k], ebar_iv + j * N_valid, N_valid, csi_list[i] + col_cursor * N_valid);
                col_cursor++;
            }
        }
    }
    int total_rows = n * N_valid;
    double *Phi_vrf = calloc(total_rows * P, sizeof(double)), *Csi_vrf = calloc(total_rows * P, sizeof(double)), *Uf_vec = calloc(total_rows, sizeof(double));
    for (int i = 0; i < n; ++i) {
        memcpy(Uf_vec + i * N_valid, uf + i * N, sizeof(double) * N_valid);
        for (int j = 0; j < n; ++j) {
            int p_cnt = row_param_counts[j];
            if (p_cnt == 0) continue;
            double *tmp_phi = calloc(N_valid * p_cnt, sizeof(double)), *tmp_csi = calloc(N_valid * p_cnt, sizeof(double));
            col_filter_scalar(&req->L.elements[i * n + j], phi_list[j], N_valid, p_cnt, tmp_phi);
            col_filter_scalar(&req->L.elements[i * n + j], csi_list[j], N_valid, p_cnt, tmp_csi);
            for (int c = 0; c < p_cnt; ++c) {
                int target_col = row_param_offsets[j] + c;
                for (int r = 0; r < N_valid; ++r) {
                    Phi_vrf[(i * N_valid + r) * P + target_col] = tmp_phi[c * N_valid + r];
                    Csi_vrf[(i * N_valid + r) * P + target_col] = tmp_csi[c * N_valid + r];
                }
            }
            free(tmp_phi); free(tmp_csi);
        }
    }
    double *Z = calloc(P * P, sizeof(double)), *Y = calloc(P, sizeof(double));
    for (int i = 0; i < P; ++i) {
        for (int r = 0; r < total_rows; ++r) {
            double csi_val = Csi_vrf[r * P + i];
            Y[i] += csi_val * Uf_vec[r];
            for (int j = 0; j < P; ++j) Z[i * P + j] += csi_val * Phi_vrf[r * P + j];
        }
    }
    solve_box_qp_cd_scalar(Z, Y, req->p_lower, req->p_upper, P, p_out);
    for (int i = 0; i < n; ++i) { free(phi_list[i]); free(csi_list[i]); }
    free(phi_list); free(csi_list); free(row_param_counts); free(row_param_offsets);
    free(uf); free(rv); free(rv_iv); free(ebar); free(ebar_iv);
    free(Phi_vrf); free(Csi_vrf); free(Uf_vec); free(Z); free(Y);
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t r = recv(fd, p, len, 0);
        if (r <= 0) return -1;
        p += r; len -= r;
    }
    return 0;
}

static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t s = send(fd, p, len, 0);
        if (s <= 0) return -1;
        p += s; len -= s;
    }
    return 0;
}

static void free_tf(DiscreteTF *tf) {
    free(tf->num); free(tf->den);
}

static int parse_payload_mem(const uint8_t *buf, size_t total_len, VrftRequest *req) {
    size_t offset = 0;
    #define READ_MEM(dst, bytes) do { \
        if (offset + (bytes) > total_len) return -1; \
        memcpy((dst), buf + offset, (bytes)); \
        offset += (bytes); \
    } while(0)

    int32_t header[3];
    READ_MEM(header, sizeof(header));
    int N = header[0], n = header[1], P = header[2];
    if (N <= 0 || n <= 0 || P <= 0) return -1;

    *req = (VrftRequest){ .n_samples = N, .n_channels = n, .total_params = P };
    req->u = malloc(sizeof(double) * N * n);
    req->y = malloc(sizeof(double) * N * n);
    req->y_iv = malloc(sizeof(double) * N * n);
    req->p_lower = malloc(sizeof(double) * P);
    req->p_upper = malloc(sizeof(double) * P);
    req->Td = (MimoTF){ n, n, calloc(n * n, sizeof(DiscreteTF)) };
    req->L = (MimoTF){ n, n, calloc(n * n, sizeof(DiscreteTF)) };
    req->c_len = calloc(n * n, sizeof(int));
    req->C_basis = calloc(n * n, sizeof(DiscreteTF *));

    READ_MEM(req->u, sizeof(double) * N * n);
    READ_MEM(req->y, sizeof(double) * N * n);
    READ_MEM(req->y_iv, sizeof(double) * N * n);

    MimoTF *mimos[2] = { &req->Td, &req->L };
    for (int m = 0; m < 2; ++m) {
        for (int i = 0; i < n * n; ++i) {
            int32_t dims[2];
            READ_MEM(dims, sizeof(dims));
            DiscreteTF *tf = &mimos[m]->elements[i];
            tf->num_len = dims[0]; tf->den_len = dims[1];
            tf->num = dims[0] > 0 ? malloc(sizeof(double) * dims[0]) : NULL;
            tf->den = dims[1] > 0 ? malloc(sizeof(double) * dims[1]) : NULL;
            if (dims[0] > 0) READ_MEM(tf->num, sizeof(double) * dims[0]);
            if (dims[1] > 0) READ_MEM(tf->den, sizeof(double) * dims[1]);
        }
    }

    for (int i = 0; i < n * n; ++i) {
        int32_t basis_cnt = 0;
        READ_MEM(&basis_cnt, sizeof(int32_t));
        req->c_len[i] = basis_cnt;
        req->C_basis[i] = basis_cnt > 0 ? calloc(basis_cnt, sizeof(DiscreteTF)) : NULL;
        for (int k = 0; k < basis_cnt; ++k) {
            int32_t dims[2];
            READ_MEM(dims, sizeof(dims));
            DiscreteTF *tf = &req->C_basis[i][k];
            tf->num_len = dims[0]; tf->den_len = dims[1];
            tf->num = dims[0] > 0 ? malloc(sizeof(double) * dims[0]) : NULL;
            tf->den = dims[1] > 0 ? malloc(sizeof(double) * dims[1]) : NULL;
            if (dims[0] > 0) READ_MEM(tf->num, sizeof(double) * dims[0]);
            if (dims[1] > 0) READ_MEM(tf->den, sizeof(double) * dims[1]);
        }
    }
    READ_MEM(req->p_lower, sizeof(double) * P);
    READ_MEM(req->p_upper, sizeof(double) * P);
    #undef READ_MEM
    return 0;
}

static void free_vrft_request(VrftRequest *req) {
    int n = req->n_channels;
    for (int i = 0; i < n * n; ++i) {
        free_tf(&req->Td.elements[i]);
        free_tf(&req->L.elements[i]);
        if (req->C_basis && req->C_basis[i]) {
            for (int k = 0; k < req->c_len[i]; ++k) free_tf(&req->C_basis[i][k]);
            free(req->C_basis[i]);
        }
    }
    free(req->Td.elements); free(req->L.elements);
    free(req->c_len); free(req->C_basis);
    free(req->u); free(req->y); free(req->y_iv);
    free(req->p_lower); free(req->p_upper);
}

static void handle_client_session(int fd, const VrftPipeline *pipeline) {
    char hdr_buf[4096];
    while (1) {
        int hdr_len = 0;
        char *body_start = NULL;
        while (hdr_len < (int)sizeof(hdr_buf) - 1) {
            ssize_t r = recv(fd, hdr_buf + hdr_len, sizeof(hdr_buf) - 1 - hdr_len, 0);
            if (r <= 0) { close(fd); return; }
            hdr_len += r;
            hdr_buf[hdr_len] = '\0';
            if ((body_start = strstr(hdr_buf, "\r\n\r\n")) != NULL) {
                body_start += 4;
                break;
            }
        }
        if (!body_start) { close(fd); return; }

        if (strncmp(hdr_buf, "GET /healthz", 12) == 0) {
            const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
            send_all(fd, resp, strlen(resp));
            continue;
        }

        char *cl_ptr = strcasestr(hdr_buf, "Content-Length:");
        size_t content_len = cl_ptr ? strtoul(cl_ptr + 15, NULL, 10) : 0;
        if (content_len == 0 || content_len > 64 * 1024 * 1024) { close(fd); return; }

        size_t already_read = hdr_len - (body_start - hdr_buf);
        uint8_t *payload = malloc(content_len);
        if (already_read > 0) memcpy(payload, body_start, already_read);
        if (already_read < content_len && recv_all(fd, payload + already_read, content_len - already_read) < 0) {
            free(payload); close(fd); return;
        }

        VrftRequest req = {0};
        if (parse_payload_mem(payload, content_len, &req) == 0) {
            double *p_opt = calloc(req.total_params, sizeof(double));
            int32_t status = vrft_design_avx2(&req, pipeline, p_opt);
            int32_t out_len = (status == 0) ? req.total_params : 0;
            size_t resp_body_bytes = sizeof(int32_t) * 2 + sizeof(double) * out_len;

            char resp_hdr[256];
            int hlen = snprintf(resp_hdr, sizeof(resp_hdr),
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                "Connection: keep-alive\r\nContent-Length: %zu\r\n\r\n", resp_body_bytes);

            send_all(fd, resp_hdr, hlen);
            send_all(fd, &status, sizeof(int32_t));
            send_all(fd, &out_len, sizeof(int32_t));
            if (out_len > 0) send_all(fd, p_opt, sizeof(double) * out_len);
            free(p_opt);
        }
        free_vrft_request(&req);
        free(payload);

        if (strcasestr(hdr_buf, "Connection: close")) { close(fd); return; }
    }
}

static void *worker_entry(void *arg) {
    ThreadPoolQueue *q = (ThreadPoolQueue *)arg;
    while (1) {
        pthread_mutex_lock(&q->lock);
        while (q->count == 0) pthread_cond_wait(&q->notify, &q->lock);
        JobItem job = q->buffer[q->head];
        q->head = (q->head + 1) % MAX_QUEUE;
        q->count--;
        pthread_mutex_unlock(&q->lock);
        handle_client_session(job.client_fd, job.pipeline);
    }
    return NULL;
}

static void run_avx2_benchmark(void) {
    int configs[3][3] = { {250, 2, 4}, {1000, 4, 4}, {2000, 6, 4} };
    VrftPipeline pipe = { .invert_model = default_stable_inversion, .solve_constrained = solve_box_qp_cd_avx2 };

    printf("====================================================================================\n");
    printf("   VRFT Algorithm Benchmark: Scalar vs. AVX2+FMA Engine (Single-Thread Pure Math)   \n");
    printf("====================================================================================\n");
    printf("%-28s | %-14s | %-14s | %-9s | %-10s\n", "Scenario (N, n, P)", "Scalar (us)", "AVX2+FMA (us)", "Speedup", "Max Diff");
    printf("------------------------------------------------------------------------------------\n");

    for (int c_idx = 0; c_idx < 3; ++c_idx) {
        int N = configs[c_idx][0], n = configs[c_idx][1], b_cnt = configs[c_idx][2];
        int P = n * b_cnt;
        VrftRequest req = { .n_samples = N, .n_channels = n, .total_params = P };
        req.u = calloc(N * n, sizeof(double));
        req.y = calloc(N * n, sizeof(double));
        req.y_iv = calloc(N * n, sizeof(double));
        req.p_lower = calloc(P, sizeof(double));
        req.p_upper = calloc(P, sizeof(double));
        req.Td = (MimoTF){ n, n, calloc(n * n, sizeof(DiscreteTF)) };
        req.L = (MimoTF){ n, n, calloc(n * n, sizeof(DiscreteTF)) };
        req.c_len = calloc(n * n, sizeof(int));
        req.C_basis = calloc(n * n, sizeof(DiscreteTF *));

        srand(42);
        for (int i = 0; i < N * n; ++i) {
            req.u[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            req.y[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            req.y_iv[i] = req.y[i] + (((double)rand() / RAND_MAX) - 0.5) * 0.05;
        }
        for (int i = 0; i < P; ++i) { req.p_lower[i] = -0.8; req.p_upper[i] = 1.2; }

        for (int i = 0; i < n; ++i) {
            DiscreteTF *td = &req.Td.elements[i * n + i];
            td->num_len = 2; td->den_len = 2;
            td->num = malloc(sizeof(double) * 2); td->num[0] = 0.0; td->num[1] = 0.4;
            td->den = malloc(sizeof(double) * 2); td->den[0] = 1.0; td->den[1] = -0.6;

            DiscreteTF *l_tf = &req.L.elements[i * n + i];
            l_tf->num_len = 1; l_tf->den_len = 2;
            l_tf->num = malloc(sizeof(double) * 1); l_tf->num[0] = 0.4;
            l_tf->den = malloc(sizeof(double) * 2); l_tf->den[0] = 1.0; l_tf->den[1] = -0.6;

            req.c_len[i * n + i] = b_cnt;
            req.C_basis[i * n + i] = calloc(b_cnt, sizeof(DiscreteTF));
            for (int k = 0; k < b_cnt; ++k) {
                DiscreteTF *cb = &req.C_basis[i * n + i][k];
                cb->num_len = 2; cb->den_len = 2;
                cb->num = malloc(sizeof(double) * 2); cb->num[0] = (k == 0) ? 1.0 : 0.0; cb->num[1] = (k == 0) ? 0.0 : 1.0;
                cb->den = malloc(sizeof(double) * 2); cb->den[0] = 1.0; cb->den[1] = -0.5 / (k + 1);
            }
        }

        double *p_ref = calloc(P, sizeof(double)), *p_avx = calloc(P, sizeof(double));
        int runs = (N <= 250) ? 1000 : 200;

        vrft_design_scalar(&req, &pipe, p_ref);
        vrft_design_avx2(&req, &pipe, p_avx);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < runs; ++r) vrft_design_scalar(&req, &pipe, p_ref);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us_scalar = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3) / runs;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < runs; ++r) vrft_design_avx2(&req, &pipe, p_avx);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us_avx = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3) / runs;

        double max_err = 0.0;
        for (int i = 0; i < P; ++i) max_err = fmax(max_err, fabs(p_ref[i] - p_avx[i]));

        char label[64];
        snprintf(label, sizeof(label), "MIMO %dx%d (N=%d, P=%d)", n, n, N, P);
        printf("%-28s | %11.2f us | %11.2f us | %8.2fx | %.2e\n", label, us_scalar, us_avx, us_scalar / us_avx, max_err);

        free(p_ref); free(p_avx);
        free_vrft_request(&req);
    }
    printf("====================================================================================\n");
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        run_avx2_benchmark();
        return 0;
    }

    const char *bind_ip = (argc > 1) ? argv[1] : "0.0.0.0";
    int port = (argc > 2) ? atoi(argv[2]) : 9090;

    VrftPipeline pipeline = {
        .invert_model = default_stable_inversion,
        .solve_constrained = solve_box_qp_cd_avx2
    };

    ThreadPoolQueue queue = { .head = 0, .tail = 0, .count = 0 };
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.notify, NULL);

    pthread_t workers[WORKER_THREADS];
    for (int i = 0; i < WORKER_THREADS; ++i) {
        pthread_create(&workers[i], NULL, worker_entry, &queue);
        pthread_detach(workers[i]);
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
    inet_pton(AF_INET, bind_ip, &addr.sin_addr);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 || listen(server_fd, 512) < 0) {
        perror("Server initialization failed");
        return 1;
    }
    printf("VRFT AVX2+FMA HTTP API Server listening on %s:%d with %d workers\n", bind_ip, port, WORKER_THREADS);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;
        pthread_mutex_lock(&queue.lock);
        if (queue.count < MAX_QUEUE) {
            queue.buffer[queue.tail] = (JobItem){ client_fd, &pipeline };
            queue.tail = (queue.tail + 1) % MAX_QUEUE;
            queue.count++;
            pthread_cond_signal(&queue.notify);
            pthread_mutex_unlock(&queue.lock);
        } else {
            pthread_mutex_unlock(&queue.lock);
            close(client_fd);
        }
    }
    return 0;
}
