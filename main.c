#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <immintrin.h>

typedef struct {
    int client_fd;
    int tau;
    float high_limit;
    float *y;
    float *u;
} __attribute__((aligned(64))) vrft_ctx_t;

// 封裝對齊記憶體分配，處理邊界與錯誤狀況
static inline void* alloc_aligned(size_t alignment, size_t size) {
    void* ptr = NULL;
    return (posix_memalign(&ptr, alignment, size) == 0) ? ptr : NULL;
}

static inline float hsum_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, 0x01);
    vlow = _mm_add_ps(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

static inline void solve_3x3_lsq(float m[3][3], float v[3], float res[3]) {
    float inv[3][3];
    inv[0][0] = m[1][1]*m[2][2] - m[1][2]*m[2][1];
    inv[0][1] = m[0][2]*m[2][1] - m[0][1]*m[2][2];
    inv[0][2] = m[0][1]*m[1][2] - m[0][2]*m[1][1];
    inv[1][0] = m[1][2]*m[2][0] - m[1][0]*m[2][2];
    inv[1][1] = m[0][0]*m[2][2] - m[0][2]*m[2][0];
    inv[1][2] = m[0][2]*m[1][0] - m[0][0]*m[1][2];
    inv[2][0] = m[1][0]*m[2][1] - m[1][1]*m[2][0];
    inv[2][1] = m[0][1]*m[2][0] - m[0][0]*m[2][1];
    inv[2][2] = m[0][0]*m[1][1] - m[0][1]*m[1][0];

    float det = m[0][0]*inv[0][0] + m[0][1]*inv[1][0] + m[0][2]*inv[2][0] + 1e-8f; 
    float inv_det = 1.0f / det;

    for (int i = 0; i < 3; ++i) {
        res[i] = (inv[i][0]*v[0] + inv[i][1]*v[1] + inv[i][2]*v[2]) * inv_det;
    }
}

static inline void compute_vrft_lsq(const float* __restrict y, const float* __restrict u, int tau, float high_limit, float res[3]) {
    (void)high_limit; 
    __m256 p11 = _mm256_setzero_ps(), p12 = _mm256_setzero_ps(), p13 = _mm256_setzero_ps();
    __m256 p22 = _mm256_setzero_ps(), p23 = _mm256_setzero_ps(), p33 = _mm256_setzero_ps();
    __m256 u1 = _mm256_setzero_ps(), u2 = _mm256_setzero_ps(), u3 = _mm256_setzero_ps();

    int limit = tau & ~7; 
    for (int i = 0; i < limit; i += 8) {
        __m256 vy = _mm256_load_ps(&y[i]);
        __m256 vu = _mm256_load_ps(&u[i]);
        
        __m256 phi1 = vy; 
        __m256 phi2 = _mm256_mul_ps(vy, _mm256_set1_ps(0.5f)); 
        __m256 phi3 = _mm256_mul_ps(vy, _mm256_set1_ps(0.333f));

        p11 = _mm256_fmadd_ps(phi1, phi1, p11);
        p12 = _mm256_fmadd_ps(phi1, phi2, p12);
        p13 = _mm256_fmadd_ps(phi1, phi3, p13);
        p22 = _mm256_fmadd_ps(phi2, phi2, p22);
        p23 = _mm256_fmadd_ps(phi2, phi3, p23);
        p33 = _mm256_fmadd_ps(phi3, phi3, p33);

        u1 = _mm256_fmadd_ps(phi1, vu, u1);
        u2 = _mm256_fmadd_ps(phi2, vu, u2);
        u3 = _mm256_fmadd_ps(phi3, vu, u3);
    }
    
    float m[3][3] = {
        {hsum_avx(p11), hsum_avx(p12), hsum_avx(p13)},
        {hsum_avx(p12), hsum_avx(p22), hsum_avx(p23)},
        {hsum_avx(p13), hsum_avx(p23), hsum_avx(p33)}
    };
    float v[3] = {hsum_avx(u1), hsum_avx(u2), hsum_avx(u3)};
    
    for (int i = limit; i < tau; ++i) {
        float p1 = y[i], p2 = y[i]*0.5f, p3 = y[i]*0.333f;
        m[0][0] += p1*p1; m[0][1] += p1*p2; m[0][2] += p1*p3;
        m[1][1] += p2*p2; m[1][2] += p2*p3; m[2][2] += p3*p3;
        v[0] += p1*u[i]; v[1] += p2*u[i]; v[2] += p3*u[i];
    }
    m[1][0] = m[0][1]; m[2][0] = m[0][2]; m[2][1] = m[1][2]; 

    solve_3x3_lsq(m, v, res);
}

void* api_router_worker(void* arg) {
    vrft_ctx_t* ctx = (vrft_ctx_t*)arg;
    char buffer[4096] = {0};
    
    if (recv(ctx->client_fd, buffer, sizeof(buffer) - 1, 0) <= 0) {
        close(ctx->client_fd);
        free(ctx);
        return NULL;
    }
    
    if (strncmp(buffer, "GET /health", 11) == 0) {
        const char* resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\": \"OK\", \"code\": 200}";
        (void)send(ctx->client_fd, resp, strlen(resp), 0);
    } 
    else if (strncmp(buffer, "POST /vrft", 10) == 0) {
        ctx->tau = 1024;
        ctx->high_limit = 100.0f;
        
        ctx->y = (float*)alloc_aligned(32, ctx->tau * sizeof(float));
        ctx->u = (float*)alloc_aligned(32, ctx->tau * sizeof(float));
        
        if (ctx->y && ctx->u) {
            memset(ctx->y, 0, ctx->tau * sizeof(float)); 
            memset(ctx->u, 0, ctx->tau * sizeof(float)); 

            float res[3] = {0};
            compute_vrft_lsq(ctx->y, ctx->u, ctx->tau, ctx->high_limit, res);
            
            char resp[256];
            snprintf(resp, sizeof(resp), 
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"Kp\": %.4f, \"Ki\": %.4f, \"Kd\": %.4f}", 
                res[0], res[1], res[2]);
            (void)send(ctx->client_fd, resp, strlen(resp), 0);
        } else {
            const char* err = "HTTP/1.1 500 Internal Server Error\r\n\r\n";
            (void)send(ctx->client_fd, err, strlen(err), 0);
        }
        
        free(ctx->y); // 利用 free(NULL) 合法特性，統一回收點
        free(ctx->u);
    }

    close(ctx->client_fd);
    free(ctx); 
    return NULL;
}

int main(void) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return EXIT_FAILURE;
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_addr.s_addr = INADDR_ANY, .sin_port = htons(80) };
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) return EXIT_FAILURE;
    if (listen(server_fd, SOMAXCONN) < 0) return EXIT_FAILURE;

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        vrft_ctx_t* ctx = (vrft_ctx_t*)alloc_aligned(64, sizeof(vrft_ctx_t)); 
        if (!ctx) {
            close(client_fd);
            continue;
        }
        ctx->client_fd = client_fd;

        pthread_t tid;
        if (pthread_create(&tid, NULL, api_router_worker, ctx) != 0) {
            close(client_fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid); 
    }
    return 0;
}
