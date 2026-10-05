#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>     // 加入 timeval
#include <immintrin.h>
#include <cjson/cJSON.h>  // 使用系統的 cJSON

typedef struct {
    int client_fd;
    int tau;
    float high_limit;
    float *y;
    float *u;
} __attribute__((aligned(64))) vrft_ctx_t;

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

// 輔助函式：發送 HTTP 回應
static void send_http_response(int fd, const char* status, const char* body) {
    char resp[1024];
    snprintf(resp, sizeof(resp), 
        "HTTP/1.1 %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n"
        "%s", 
        status, strlen(body), body);
    (void)send(fd, resp, strlen(resp), 0);
}

void* api_router_worker(void* arg) {
    vrft_ctx_t* ctx = (vrft_ctx_t*)arg;

    // 加入 Socket Timeout，避免讀取被惡意或慢速請求掛死
    struct timeval tv;
    tv.tv_sec = 3;  // 3秒超時
    tv.tv_usec = 0;
    setsockopt(ctx->client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

    char buffer[8192] = {0};
    int total_received = 0;
    char* header_end = NULL;

    // 迴圈讀取，確切保證拿到完整的 HTTP Header（找到 \r\n\r\n 為止）
    while (total_received < sizeof(buffer) - 1) {
        int r = recv(ctx->client_fd, buffer + total_received, sizeof(buffer) - 1 - total_received, 0);
        if (r <= 0) break;
        total_received += r;
        buffer[total_received] = '\0'; // 補上結尾以便字串搜尋

        header_end = strstr(buffer, "\r\n\r\n");
        if (header_end) {
            break; 
        }
    }

    // 若成功拿到完整 Header 才開始處理
    if (header_end) {
        if (strncmp(buffer, "GET /health", 11) == 0) {
            send_http_response(ctx->client_fd, "200 OK", "{\"status\": \"OK\", \"code\": 200}");
        } 
        else if (strncmp(buffer, "POST /vrft", 10) == 0) {
            int content_length = 0;
            // 找出 Content-Length (容錯大小寫)
            char* cl_ptr = strstr(buffer, "Content-Length: ");
            if (!cl_ptr) cl_ptr = strstr(buffer, "content-length: ");
            
            if (cl_ptr) {
                content_length = atoi(cl_ptr + 16);
            }

            if (content_length > 0) {
                header_end += 4; // 跳過 \r\n\r\n
                int header_size = header_end - buffer;
                int body_received = total_received - header_size;
                
                char* body_buffer = (char*)malloc(content_length + 1);
                if (body_buffer) {
                    memcpy(body_buffer, header_end, body_received);
                    int body_read = body_received;
                    
                    // 迴圈把 Body(JSON) 剩下的資料徹底讀滿
                    while (body_read < content_length) {
                        int r = recv(ctx->client_fd, body_buffer + body_read, content_length - body_read, 0);
                        if (r <= 0) break;
                        body_read += r;
                    }
                    body_buffer[body_read] = '\0'; 

                    // 使用 cJSON 解析
                    cJSON *json = cJSON_Parse(body_buffer);
                    if (json) {
                        cJSON *j_tau = cJSON_GetObjectItem(json, "tau");
                        cJSON *j_hl = cJSON_GetObjectItem(json, "high_limit");
                        cJSON *j_y = cJSON_GetObjectItem(json, "y");
                        cJSON *j_u = cJSON_GetObjectItem(json, "u");

                        if (cJSON_IsNumber(j_tau) && cJSON_IsNumber(j_hl) && 
                            cJSON_IsArray(j_y) && cJSON_IsArray(j_u)) {
                            
                            ctx->tau = j_tau->valueint;
                            ctx->high_limit = (float)j_hl->valuedouble;
                            
                            int y_size = cJSON_GetArraySize(j_y);
                            int u_size = cJSON_GetArraySize(j_u);

                            if (ctx->tau > 0 && y_size >= ctx->tau && u_size >= ctx->tau) {
                                ctx->y = (float*)alloc_aligned(32, ctx->tau * sizeof(float));
                                ctx->u = (float*)alloc_aligned(32, ctx->tau * sizeof(float));
                                
                                if (ctx->y && ctx->u) {
                                    cJSON *item_y = j_y->child;
                                    cJSON *item_u = j_u->child;
                                    for (int i = 0; i < ctx->tau; i++) {
                                        ctx->y[i] = item_y ? (float)item_y->valuedouble : 0.0f;
                                        ctx->u[i] = item_u ? (float)item_u->valuedouble : 0.0f;
                                        if (item_y) item_y = item_y->next;
                                        if (item_u) item_u = item_u->next;
                                    }

                                    // AVX 運算
                                    float res[3] = {0};
                                    compute_vrft_lsq(ctx->y, ctx->u, ctx->tau, ctx->high_limit, res);
                                    
                                    char resp_body[256];
                                    snprintf(resp_body, sizeof(resp_body), 
                                        "{\"Kp\": %.4f, \"Ki\": %.4f, \"Kd\": %.4f}", 
                                        res[0], res[1], res[2]);
                                    
                                    send_http_response(ctx->client_fd, "200 OK", resp_body);

                                } else {
                                    send_http_response(ctx->client_fd, "500 Internal Server Error", "{\"error\": \"Alloc Failed\"}");
                                }
                            } else {
                                send_http_response(ctx->client_fd, "400 Bad Request", "{\"error\": \"Array too small\"}");
                            }
                        } else {
                            send_http_response(ctx->client_fd, "400 Bad Request", "{\"error\": \"Invalid JSON fields\"}");
                        }
                        cJSON_Delete(json);
                    } else {
                        send_http_response(ctx->client_fd, "400 Bad Request", "{\"error\": \"Invalid JSON format\"}");
                    }
                    free(body_buffer);
                }
            } else {
                send_http_response(ctx->client_fd, "400 Bad Request", "{\"error\": \"Missing Content-Length\"}");
            }
        } else {
            send_http_response(ctx->client_fd, "404 Not Found", "{\"error\": \"Endpoint not found\"}");
        }
    }
    
    // SHUT_WR 會告知 "我資料送完了(送出 FIN)"
    shutdown(ctx->client_fd, SHUT_WR);

    // 可能殘留還在送進來的封包吃乾淨 (排空緩衝區)
    char discard[4096];
    while (recv(ctx->client_fd, discard, sizeof(discard), 0) > 0) {
        // do nothing
    }

    close(ctx->client_fd);
    if (ctx->y) free(ctx->y);
    if (ctx->u) free(ctx->u);
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
        ctx->y = NULL;
        ctx->u = NULL;

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
