#ifndef CORE_H
#define CORE_H

#include <math.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "font.h"

#define FLT_MAX  3.402823e+38f
#define FLT_MIN 1e-36f

#define M_GRAV 6.67430e-11f
#define M_G 9.80665f
#define M_K 9e+9f
#define M_MAGNITIC 1.25663706e-6f

#define GROUND_MU_TREN 0.3f

#define SIGN(x) ((x > 0) - (x < 0))

typedef struct {
    float x;
    float y;
} Vector2f;

typedef struct {
    short x;
    short y;
} Vector2s;

typedef struct {
    int x;
    int y;
} Vector2i;

typedef struct{
    Vector2f speed;
    float scale;
    float x[4];
    float y[4];
    float width;
    float height;
    uint32_t color;
    float deg;
    short refraction[2];
    float mass;
    Vector2f max_speed;
} Rect;

typedef struct{
    Vector2f speed;
    unsigned short scale;
    Vector2f loc;
    wchar_t text[32];
    uint32_t color;
    unsigned int len;
    short refraction[2];
    // float mass;
    // Vector2f max_speed;
} Text;

typedef struct {
    float x[2];
    float y[2];
    Vector2f speed;
    uint32_t color;
    float deg;
    short refraction[2];
    float mass;
    Vector2f max_speed;
} Line;

typedef struct {
    Vector2f loc;
    Vector2f speed;
    uint32_t color;
    short refraction[2];
    float mass;
    Vector2f max_speed;
} Pixel;

typedef struct {
    uint16_t width;
    uint16_t height;
    uint32_t *pixels;
    Vector2f loc;
    Vector2f speed;
    short refraction[2];
    float mass;
    Vector2f max_speed;
} TGA_sprite;

typedef struct{
    Vector2f speed;
    float scale;
    Vector2f loc;
    Vector2f screen_loc;
    float r;
    uint32_t color;
    short refraction[2];
    float mass;
    Vector2f max_speed;
    Vector2s move;
    float q;
    float elasticity;
} Circle;

typedef struct{
    Vector2f speed;
    float scale;
    float x[3];
    float y[3];
    uint32_t color;
    float deg;
    short refraction[2];
    float mass;
    Vector2f max_speed;
} Triangle;

typedef struct {
    Vector2f loc;
    Vector2f speed;
    Vector2f max_speed;
    float scale;
    float zoom;
    float deg;
    float rad;
    Vector2s move;
} Camera;

#pragma pack(push, 1)
typedef struct {
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];

    char     subchunk1_id[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;

    char     subchunk2_id[4];
    uint32_t subchunk2_size;
} WavHeader;

typedef struct {
    uint8_t  id_length;
    uint8_t  color_map_type;
    uint8_t  image_type;
    uint16_t color_map_origin;
    uint16_t color_map_length;
    uint8_t  color_map_depth;
    uint16_t x_origin;
    uint16_t y_origin;
    uint16_t width;
    uint16_t height;
    uint8_t  bits_per_pixel;
    uint8_t  image_descriptor;
} TGAHeader;
#pragma pack(pop)

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

static inline float max_vector(float *arr, short n){
    float maximum = -INFINITY;

    short i = 0;

    for (; i < (n & ~1); i += 2){
        maximum = fmaxf(maximum, fmaxf(arr[i], arr[i + 1]));
    }

    return (n & 1) ? fmaxf(maximum, arr[n-1]) : maximum;
}

static inline float min_vector(float *arr, short n){
    float minimum = INFINITY;

    short i = 0;

    for (; i < (n & ~1); i += 2){
        minimum = fminf(minimum, fminf(arr[i], arr[i + 1]));
    }

    return (n & 1) ? fminf(minimum, arr[n-1]) : minimum;
}

static inline void vector_add_scal(float *arr, short n, float sc){
    if (sc == 0.0f) return;
    for (short i = 0; i < n; i++){
        arr[i] += sc;
    }
}

static inline float reciprocal(float z) {
    __m128 reg = _mm_set_ss(z);
    reg = _mm_rcp_ss(reg);
    return _mm_cvtss_f32(reg);
}

static inline void apply_volume(short *buffer, uint32_t bytes_count, float volume) {
    uint32_t samples_count = bytes_count / sizeof(short);

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < samples_count; i++) {
        buffer[i] = (short)max(-32767, min((buffer[i] * volume), 32767));
    }
}

static inline uint32_t blend_pixels(uint32_t src, uint32_t dst) {
    uint32_t alpha = (src >> 24) & 0xFF;

    uint32_t src_rb = src & 0x00FF00FF;
    uint32_t src_g = src & 0x0000FF00;

    uint32_t dst_rb = dst & 0x00FF00FF;
    uint32_t dst_g = dst & 0x0000FF00;

    uint32_t blend_rb = dst_rb + (((src_rb - dst_rb) * alpha) >> 8);
    uint32_t blend_g = dst_g + (((src_g - dst_g) * alpha) >> 8);

    return (blend_rb & 0x00FF00FF) | (blend_g & 0x0000FF00);
}

static inline __m256i blend_pixels_avx(__m256i src_v, __m256i dst_v){
    /* _mm256_slli_epiXX << */
    /* _mm256_srli_epiXX >> */
    /* _mm256_srai_epiXX >> */

    __m256i alpha_v = _mm256_srli_epi32(src_v, 24);

    __m256i mask_rb = _mm256_set1_epi32(0x00FF00FF);
    __m256i mask_g = _mm256_set1_epi32(0x0000FF00);

    __m256i src_rb = _mm256_and_si256(src_v, mask_rb);
    __m256i src_g = _mm256_and_si256(src_v, mask_g);

    __m256i dst_rb = _mm256_and_si256(dst_v, mask_rb);
    __m256i dst_g = _mm256_and_si256(dst_v, mask_g);

    __m256i rb_sub = _mm256_sub_epi32(src_rb, dst_rb);
    __m256i g_sub = _mm256_sub_epi32(src_g, dst_g);

    __m256i rb_mul = _mm256_mullo_epi32(rb_sub, alpha_v);
    __m256i g_mul = _mm256_mullo_epi32(g_sub, alpha_v);

    __m256i rb_bit = _mm256_srli_epi32(rb_mul, 8);
    __m256i g_bit = _mm256_srli_epi32(g_mul, 8);

    __m256i blend_rb = _mm256_add_epi32(dst_rb, rb_bit);
    __m256i blend_g = _mm256_add_epi32(dst_g, g_bit);

    __m256i a1 = _mm256_and_si256(blend_rb, mask_rb);
    __m256i a2 = _mm256_and_si256(blend_g, mask_g);

    return _mm256_or_si256(a1, a2);
}

static inline __m256i rand_epi32(__m256i *seed_vec, __m256i max_v) {

    __m256i seed_1 = _mm256_slli_epi32(*seed_vec, 1);
    *seed_vec = _mm256_xor_si256(*seed_vec, seed_1);

    __m256i seed_2 = _mm256_srli_epi32(*seed_vec, 3);
    *seed_vec = _mm256_xor_si256(*seed_vec, seed_2);

    __m256i rand_bits = _mm256_srli_epi32(*seed_vec, 16);
    rand_bits = _mm256_and_si256(rand_bits, _mm256_set1_epi32(0x7FFF));

    __m256i res = _mm256_mullo_epi32(rand_bits, max_v);

    return _mm256_srli_epi32(res, 15);
}

static inline void shum(short p, short width, short height, uint32_t *framebuffer){
    p = (p + 1) & 100;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; i += 99) {
        for (short x = 0; x < p; x++){
            short a = rand() % 101;
            framebuffer[i + a] = (uint32_t)rand();
        }
    }
}

static inline void shum_avx(short p, int pixels, uint32_t *framebuffer){
    __m256i seed_vec = _mm256_setr_epi32(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
    __m256i max_v = _mm256_set1_epi32(100);

    int iters = pixels & ~7;
    int i = 0;

    __m256i mask = _mm256_set1_epi32(p);

    for (; i < iters; i += 8) {
        _mm256_maskstore_epi32((int*)(&framebuffer[i]), _mm256_cmpgt_epi32(mask, rand_epi32(&seed_vec, max_v)), seed_vec);
    }

    for (; i < pixels; i++) {
        framebuffer[i] = p > rand() % 101 ? (uint32_t)rand() : framebuffer[i];
    }
}

static inline void clear_screen(uint32_t color, short width, short height, uint32_t *framebuffer) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; i++) {
        framebuffer[i] = color;
    }
}

static inline void clear_screen_avx(uint32_t color, int pixels, uint32_t *framebuffer) {
    __m256i vec_color = _mm256_set1_epi32(color);

    int iters = pixels & ~7;
    int i = 0;

    for (; i < iters; i += 8) {
        _mm256_stream_si256((__m256i*)&framebuffer[i], vec_color);
    }

    for (; i < pixels; i++) {
        framebuffer[i] = color;
    }
}

static inline void memcpy_avx_epi32(uint32_t *dst, uint32_t *src, int size){
    int iters = size & ~7;
    int i = 0;

    for (; i < iters; i += 8) {
        _mm256_stream_si256((__m256i*)&dst[i], _mm256_load_si256((__m256i*)&src[i]));
    }

    for (; i < size; i++) {
        dst[i] = src[i];
    }
}

static inline void draw_pixel(int x, int y, uint32_t color, short width, short height, uint32_t *framebuffer) {
    if (x >= 0 && x < width && y >= 0 && y < height) framebuffer[y * width + x] = color;
}

static inline void draw_pixel_glass(int x, int y, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
    int index = y * width + x;
    if ((index < width * height - 1) && (fb[index] == 0)){
        framebuffer[index] = blend_pixels(color, framebuffer[index]);
        fb[index] = 1;
    }
}

static inline void draw_pixel_refraction(int x, int y, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    if (x >= 0 && x < width && y >= 0 && y < height) buffer[y * width + x] = blend_pixels(color, framebuffer[min((y + arr[1]) * width + x + arr[0], width * height - 1)]);
}

/* линии начало */

static inline void draw_line_avx(int x0, int x1, int y, uint32_t color, short width, short height, uint32_t *framebuffer){
    if (y < 0 || y >= height) return;

    x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

    int rows = y * width;

    int max_x = max(x0, x1);
    int min_x = min(x0, x1);

    int iters = min_x + ((max_x - min_x) & ~7);

    int x = min_x;

    __m256i vec_color = _mm256_set1_epi32(color);

    for (; x < iters; x += 8){
        _mm256_storeu_si256((__m256i*)&framebuffer[rows + x], vec_color);
    }

    for (; x < max_x; x++){
        framebuffer[rows + x] = color;
    }
}

static inline void draw_line_avx_glass(int x0, int x1, int y, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    if (y < 0 || y >= height) return;

    x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

    int rows = y * width;

    int max_x = max(x0, x1);
    int min_x = min(x0, x1);

    int iters = min_x + ((max_x - min_x) & ~7);

    int x = min_x;

    __m256i v_zero = _mm256_setzero_si256();
    __m128i v_ones = _mm_set1_epi8(1);
    __m256i vec_color = _mm256_set1_epi32(color);

    for (; x < iters; x += 8){
        int index = rows + x;

        __m256i fr_b_v = _mm256_load_si256((__m256i*)&framebuffer[index]);
        __m256i color_v = blend_pixels_avx(vec_color, fr_b_v);

        __m256i fb_v = _mm256_setr_epi32(fb[index], fb[index + 1], fb[index + 2], fb[index + 3], fb[index + 4], fb[index + 5], fb[index + 6], fb[index + 7]);
        __m256i mask = _mm256_cmpeq_epi32(fb_v, v_zero);

        _mm256_maskstore_epi32((int*)(&framebuffer[index]), mask, color_v);

        _mm_storel_epi64((__m128i*)&fb[index], v_ones);
    }

    for (; x < max_x; x++){
        int index = rows + x;

        if (fb[index] == 0) {
            framebuffer[index] = blend_pixels(color, framebuffer[index]);
            fb[index] = 1;
        }
    }
}

static inline void draw_line_avx_refraction(int x0, int x1, int y, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    if (y < 0 || y >= height) return;

    x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

    int rows = y * width;

    int max_x = max(x0, x1);
    int min_x = min(x0, x1);

    int iters = min_x + ((max_x - min_x) & ~7);

    int len = width * height - 1;
    int rows_r = arr[1] * width;

    int x = min_x;

    __m256i vec_color = _mm256_set1_epi32(color);

    for (; x < iters; x += 8){
        __m256i fr_b_v = _mm256_loadu_si256((__m256i*)&framebuffer[min(rows + x + rows_r + arr[0], len - 8)]);
        __m256i color_v = blend_pixels_avx(vec_color, fr_b_v);
        _mm256_storeu_si256((__m256i*)&buffer[rows + x], color_v);
    }

    for (; x < max_x; x++){
        buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + x + rows_r + arr[0], len)]);
    }
}

//Алгоритм Брезенхэма
static inline void draw_line(int x0, int y0, int x1, int y1, uint32_t color, short width, short height, uint32_t *framebuffer) {
    if (x1 == x0) {

        if (x0 < 0 || x0 >= width) return;

        y0 = y0 < 0 ? 0 : (y0 >= height ? height - 1 : y0);
        y1 = y1 < 0 ? 0 : (y1 >= height ? height - 1 : y1);

        int rows = min(y0, y1) * width;

        for (int y = min(y0, y1); y < max(y0, y1); y++){
            framebuffer[rows + x0] = color;
            rows += width;
        }
        return;

    }else if (y1 == y0) {

        if (y0 < 0 || y0 >= height) return;

        x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

        int rows = y0 * width;

        for (int x = min(x0, x1); x < max(x0, x1); x++){
            framebuffer[rows + x] = color;
        }

        return;

    }else{

        float k = (float)(y1 - y0) / (float)(x1 - x0);
        float b = (float)y0 - k * (float)x0;

        if (x0 < 0) {x0 = 0; y0 = b;}
        else if (x0 >= width) {x0 = width - 1; y0 = k * (float)x0 + b;}

        if (y0 < 0) {y0 = 0; x0 = -b / k;}
        else if (y0 >= height) {y0 = height - 1; x0 = ((float)y0 - b) / k;}

        if (x1 < 0) { x1 = 0; y1 = b; }
        else if (x1 >= width) {x1 = width - 1; y1 = k * (float)x1 + b;}

        if (y1 < 0) {y1 = 0; x1 = -b / k;}
        else if (y1 >= height) {y1 = height - 1; x1 = ((float)y1 - b) / k;}

        if (x0 < 0 || x0 >= width || x1 < 0 || x1 >= width || y0 < 0 || y0 >= height || y1 < 0 || y1 >= height) return;
    }

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (x0 != x1 || y0 != y1) {
        framebuffer[y0 * width + x0] = color;

        e2 = 2 * err;
        if (e2 >= dy) {err += dy; x0 += sx;}
        if (e2 <= dx) {err += dx; y0 += sy;}
    }
}

static inline void draw_line_glass(int x0, int y0, int x1, int y1, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
    if (x1 == x0) {

        if (x0 < 0 || x0 >= width) return;

        y0 = y0 < 0 ? 0 : (y0 >= height ? height - 1 : y0);
        y1 = y1 < 0 ? 0 : (y1 >= height ? height - 1 : y1);

        int rows = min(y0, y1) * width;

        for (int y = min(y0, y1); y < max(y0, y1); y++){
            int index = rows + x0;
            if (fb[index] == 0) {
                framebuffer[index] = blend_pixels(color, framebuffer[index]);
                fb[index] = 1;
            }
            rows += width;
        }
        return;

    }else if (y1 == y0) {

        if (y0 < 0 || y0 >= height) return;

        x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

        int rows = y0 * width;

        for (int x = min(x0, x1); x < max(x0, x1); x++){
            int index = rows + x;
            if (fb[index] == 0) {
                framebuffer[index] = blend_pixels(color, framebuffer[index]);
                fb[index] = 1;
            }
        }

        return;

    }else{

        float k = (float)(y1 - y0) / (float)(x1 - x0);
        float b = (float)y0 - k * (float)x0;

        if (x0 < 0) {x0 = 0; y0 = b;}
        else if (x0 >= width) {x0 = width - 1; y0 = k * (float)x0 + b;}

        if (y0 < 0) {y0 = 0; x0 = -b / k;}
        else if (y0 >= height) {y0 = height - 1; x0 = ((float)y0 - b) / k;}

        if (x1 < 0) { x1 = 0; y1 = b; }
        else if (x1 >= width) {x1 = width - 1; y1 = k * (float)x1 + b;}

        if (y1 < 0) {y1 = 0; x1 = -b / k;}
        else if (y1 >= height) {y1 = height - 1; x1 = ((float)y1 - b) / k;}

        if (x0 < 0 || x0 >= width || x1 < 0 || x1 >= width || y0 < 0 || y0 >= height || y1 < 0 || y1 >= height) return;
    }

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (x0 != x1 || y0 != y1) {
        int index = y0 * width + x0;
        if (fb[index] == 0) {
            framebuffer[index] = blend_pixels(color, framebuffer[index]);
            fb[index] = 1;
        }

        e2 = 2 * err;
        if (e2 >= dy) {err += dy; x0 += sx;}
        if (e2 <= dx) {err += dx; y0 += sy;}
    }
}

static inline void draw_line_refraction(int x0, int y0, int x1, int y1, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int len = width * height - 1;
    int rows_r = arr[1] * width;

    if (x1 == x0) {

        if (x0 < 0 || x0 >= width) return;

        y0 = y0 < 0 ? 0 : (y0 >= height ? height - 1 : y0);
        y1 = y1 < 0 ? 0 : (y1 >= height ? height - 1 : y1);

        int rows = min(y0, y1) * width;

        for (int y = min(y0, y1); y < max(y0, y1); y++){
            buffer[rows + x0] = blend_pixels(color, framebuffer[min(rows + rows_r + x0 + arr[0], len)]);
            rows += width;
        }
        return;

    }else if (y1 == y0) {

        if (y0 < 0 || y0 >= height) return;

        x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
        x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);

        int rows = y0 * width;

        for (int x = min(x0, x1); x < max(x0, x1); x++){
            buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);
        }

        return;

    }else{

        float k = (float)(y1 - y0) / (float)(x1 - x0);
        float b = (float)y0 - k * (float)x0;

        if (x0 < 0) {x0 = 0; y0 = b;}
        else if (x0 >= width) {x0 = width - 1; y0 = k * (float)x0 + b;}

        if (y0 < 0) {y0 = 0; x0 = -b / k;}
        else if (y0 >= height) {y0 = height - 1; x0 = ((float)y0 - b) / k;}

        if (x1 < 0) { x1 = 0; y1 = b; }
        else if (x1 >= width) {x1 = width - 1; y1 = k * (float)x1 + b;}

        if (y1 < 0) {y1 = 0; x1 = -b / k;}
        else if (y1 >= height) {y1 = height - 1; x1 = ((float)y1 - b) / k;}

        if (x0 < 0 || x0 >= width || x1 < 0 || x1 >= width || y0 < 0 || y0 >= height || y1 < 0 || y1 >= height) return;
    }

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (x0 != x1 || y0 != y1) {
        buffer[y0 * width + x0] = blend_pixels(color, framebuffer[min(y0 * width + rows_r + x0 + arr[0], len)]);

        e2 = 2 * err;
        if (e2 >= dy) {err += dy; x0 += sx;}
        if (e2 <= dx) {err += dx; y0 += sy;}
    }
}

/* триугольники начало */

static inline void draw_triangle_refraction(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = min(x0, min(x1, x2));
    int x_max = max(x0, max(x1, x2));
    int y_min = min(y0, min(y1, y2));
    int y_max = max(y0, max(y1, y2));

    y_max = min(height, y_max);
    y_min = max(0, y_min);
    x_max = min(width, x_max);
    x_min = max(0, x_min);

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int d1, d2, d3, start_d1, start_d2, start_d3;

    start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;
    int len = width * height - 1;
    int rows_r = arr[1] * width;

    for (int y = y_min; y < y_max; y++){
        d1 = start_d1;
        d2 = start_d2;
        d3 = start_d3;

        for (int x = x_min; x < x_max; x++){

            if (d1 >= 0 && d2 >= 0 && d3 >= 0) {
                buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);
            }

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }
        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;

        rows += width;
    }
}

static inline void draw_triangle_glass(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = min(x0, min(x1, x2));
    int x_max = max(x0, max(x1, x2));
    int y_min = min(y0, min(y1, y2));
    int y_max = max(y0, max(y1, y2));

    y_max = min(height, y_max);
    y_min = max(0, y_min);
    x_max = min(width, x_max);
    x_min = max(0, x_min);

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int d1, d2, d3, start_d1, start_d2, start_d3;

    start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;

    for (int y = y_min; y < y_max; y++){
        d1 = start_d1;
        d2 = start_d2;
        d3 = start_d3;

        for (int x = x_min; x < x_max; x++){

            int index = rows + x;

            if ((d1 >= 0 && d2 >= 0 && d3 >= 0) && fb[index] == 0) {
                framebuffer[index] = blend_pixels(color, framebuffer[index]);
                fb[index] = 1;
            }

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }
        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;

        rows += width;
    }
}

static inline void draw_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short width, short height, uint32_t *framebuffer) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = min(x0, min(x1, x2));
    int x_max = max(x0, max(x1, x2));
    int y_min = min(y0, min(y1, y2));
    int y_max = max(y0, max(y1, y2));

    y_max = min(height, y_max);
    y_min = max(0, y_min);
    x_max = min(width, x_max);
    x_min = max(0, x_min);

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int d1, d2, d3, start_d1, start_d2, start_d3;

    start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;

    for (int y = y_min; y < y_max; y++){
        d1 = start_d1;
        d2 = start_d2;
        d3 = start_d3;

        for (int x = x_min; x < x_max; x++){

            if (d1 >= 0 && d2 >= 0 && d3 >= 0) {
                framebuffer[rows + x] = color;
            }

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }
        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;

        rows += width;
    }
}

static inline void draw_triangle_avx(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short width, short height, uint32_t *framebuffer) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = max(0, min(x0, min(x1, x2)));
    int x_max = min(width, max(x0, max(x1, x2)));
    int y_min = max(0, min(y0, min(y1, y2)));
    int y_max = min(height, max(y0, max(y1, y2)));

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    int start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    int start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;
    int iters = x_min + ((x_max - x_min) & ~7);

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_color = _mm256_set1_epi32(color);

    __m256i v_sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    __m256i v_step1 = _mm256_set1_epi32(dE1_x * 8);
    __m256i v_step2 = _mm256_set1_epi32(dE2_x * 8);
    __m256i v_step3 = _mm256_set1_epi32(dE3_x * 8);

    __m256i v_dE1_x = _mm256_set1_epi32(dE1_x);
    __m256i v_dE2_x = _mm256_set1_epi32(dE2_x);
    __m256i v_dE3_x = _mm256_set1_epi32(dE3_x);

    for (int y = y_min; y < y_max; y++) {
        __m256i v_d1 = _mm256_add_epi32(_mm256_set1_epi32(start_d1), _mm256_mullo_epi32(v_dE1_x, v_sequence));
        __m256i v_d2 = _mm256_add_epi32(_mm256_set1_epi32(start_d2), _mm256_mullo_epi32(v_dE2_x, v_sequence));
        __m256i v_d3 = _mm256_add_epi32(_mm256_set1_epi32(start_d3), _mm256_mullo_epi32(v_dE3_x, v_sequence));

        int x = x_min;

        for (; x < iters; x += 8) {
            __m256i mask_eq1  = _mm256_cmpeq_epi32(v_d1, v_zero); // ==
            __m256i mask_eq2  = _mm256_cmpeq_epi32(v_d2, v_zero);
            __m256i mask_eq3  = _mm256_cmpeq_epi32(v_d3, v_zero);

            __m256i mask_pos1 = _mm256_cmpgt_epi32(v_d1, v_zero); // >
            __m256i mask_ge1  = _mm256_or_si256(mask_pos1, mask_eq1); // or | ||

            __m256i mask_pos2 = _mm256_cmpgt_epi32(v_d2, v_zero);
            __m256i mask_ge2  = _mm256_or_si256(mask_pos2, mask_eq2);

            __m256i mask_pos3 = _mm256_cmpgt_epi32(v_d3, v_zero);
            __m256i mask_ge3  = _mm256_or_si256(mask_pos3, mask_eq3);

            __m256i mask_cw = _mm256_and_si256(_mm256_and_si256(mask_ge1, mask_ge2), mask_ge3); // and & &&

            _mm256_maskstore_epi32((int*)(&framebuffer[rows + x]), mask_cw, v_color);

            v_d1 = _mm256_add_epi32(v_d1, v_step1);
            v_d2 = _mm256_add_epi32(v_d2, v_step2);
            v_d3 = _mm256_add_epi32(v_d3, v_step3);
        }

        int d1 = start_d1 + (x - x_min) * dE1_x;
        int d2 = start_d2 + (x - x_min) * dE2_x;
        int d3 = start_d3 + (x - x_min) * dE3_x;

        for (; x < x_max; x++) {
            if ((d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0)) framebuffer[rows + x] = color;

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }

        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;
        rows += width;
    }
}

static inline void draw_triangle_avx_glass(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = max(0, min(x0, min(x1, x2)));
    int x_max = min(width, max(x0, max(x1, x2)));
    int y_min = max(0, min(y0, min(y1, y2)));
    int y_max = min(height, max(y0, max(y1, y2)));

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    int start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    int start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;
    int iters = x_min + ((x_max - x_min) & ~7);

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_color = _mm256_set1_epi32(color);

    __m256i v_sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    __m256i v_step1 = _mm256_set1_epi32(dE1_x * 8);
    __m256i v_step2 = _mm256_set1_epi32(dE2_x * 8);
    __m256i v_step3 = _mm256_set1_epi32(dE3_x * 8);

    __m256i v_dE1_x = _mm256_set1_epi32(dE1_x);
    __m256i v_dE2_x = _mm256_set1_epi32(dE2_x);
    __m256i v_dE3_x = _mm256_set1_epi32(dE3_x);

    for (int y = y_min; y < y_max; y++) {
        __m256i v_d1 = _mm256_add_epi32(_mm256_set1_epi32(start_d1), _mm256_mullo_epi32(v_dE1_x, v_sequence));
        __m256i v_d2 = _mm256_add_epi32(_mm256_set1_epi32(start_d2), _mm256_mullo_epi32(v_dE2_x, v_sequence));
        __m256i v_d3 = _mm256_add_epi32(_mm256_set1_epi32(start_d3), _mm256_mullo_epi32(v_dE3_x, v_sequence));

        int x = x_min;

        for (; x < iters; x += 8) {
            int index = rows + x;

            __m256i mask_eq1  = _mm256_cmpeq_epi32(v_d1, v_zero); // ==
            __m256i mask_eq2  = _mm256_cmpeq_epi32(v_d2, v_zero);
            __m256i mask_eq3  = _mm256_cmpeq_epi32(v_d3, v_zero);

            __m256i mask_pos1 = _mm256_cmpgt_epi32(v_d1, v_zero); // >=
            __m256i mask_ge1  = _mm256_or_si256(mask_pos1, mask_eq1); // or | ||

            __m256i mask_pos2 = _mm256_cmpgt_epi32(v_d2, v_zero);
            __m256i mask_ge2  = _mm256_or_si256(mask_pos2, mask_eq2);

            __m256i mask_pos3 = _mm256_cmpgt_epi32(v_d3, v_zero);
            __m256i mask_ge3  = _mm256_or_si256(mask_pos3, mask_eq3);

            __m256i mask_cw = _mm256_and_si256(_mm256_and_si256(mask_ge1, mask_ge2), mask_ge3);


            __m256i fr_b_v = _mm256_load_si256((__m256i*)&framebuffer[index]);

            __m256i color_v = blend_pixels_avx(v_color, fr_b_v);

            __m256i fb_v = _mm256_setr_epi32(fb[index], fb[index + 1], fb[index + 2], fb[index + 3], fb[index + 4], fb[index + 5], fb[index + 6], fb[index + 7]);

            __m256i finisf_mask = _mm256_cmpeq_epi32(fb_v, v_zero);

            __m256i final_mask = _mm256_and_si256(mask_cw, finisf_mask);

            _mm256_maskstore_epi32((int*)(&framebuffer[index]), final_mask, color_v);

            __m256i pack_16 = _mm256_packs_epi32(final_mask, final_mask);
            __m256i perm = _mm256_permute4x64_epi64(pack_16, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i bytes64 = _mm_packs_epi16(_mm256_castsi256_si128(perm), _mm256_castsi256_si128(perm));

            _mm_storel_epi64((__m128i*)&fb[index], bytes64);

            v_d1 = _mm256_add_epi32(v_d1, v_step1);
            v_d2 = _mm256_add_epi32(v_d2, v_step2);
            v_d3 = _mm256_add_epi32(v_d3, v_step3);
        }

        int d1 = start_d1 + (x - x_min) * dE1_x;
        int d2 = start_d2 + (x - x_min) * dE2_x;
        int d3 = start_d3 + (x - x_min) * dE3_x;

        for (; x < x_max; x++) {
            int index = rows + x;

            if (((d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0)) && fb[index] == 0) {
                framebuffer[index] = blend_pixels(color, framebuffer[index]);
                fb[index] = 1;
            }

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }

        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;
        rows += width;
    }
}

static inline void draw_triangle_avx_refraction(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {

    if ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0) {

        int temp_x = x1;
        int temp_y = y1;

        x1 = x2;
        y1 = y2;

        x2 = temp_x;
        y2 = temp_y;
    }

    int x_min = max(0, min(x0, min(x1, x2)));
    int x_max = min(width, max(x0, max(x1, x2)));
    int y_min = max(0, min(y0, min(y1, y2)));
    int y_max = min(height, max(y0, max(y1, y2)));

    int dE1_x = y0 - y1;
    int dE2_x = y1 - y2;
    int dE3_x = y2 - y0;

    int dE1_y = x0 - x1;
    int dE2_y = x1 - x2;
    int dE3_y = x2 - x0;

    int start_d1 = (x_min - x1) * dE1_x - dE1_y * (y_min - y1);
    int start_d2 = (x_min - x2) * dE2_x - dE2_y * (y_min - y2);
    int start_d3 = (x_min - x0) * dE3_x - dE3_y * (y_min - y0);

    int rows = y_min * width;
    int iters = x_min + ((x_max - x_min) & ~7);

    int len = width * height - 1;
    int rows_r = arr[1] * width;

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_color = _mm256_set1_epi32(color);

    __m256i v_sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    __m256i v_step1 = _mm256_set1_epi32(dE1_x * 8);
    __m256i v_step2 = _mm256_set1_epi32(dE2_x * 8);
    __m256i v_step3 = _mm256_set1_epi32(dE3_x * 8);

    __m256i v_dE1_x = _mm256_set1_epi32(dE1_x);
    __m256i v_dE2_x = _mm256_set1_epi32(dE2_x);
    __m256i v_dE3_x = _mm256_set1_epi32(dE3_x);

    for (int y = y_min; y < y_max; y++) {
        __m256i v_d1 = _mm256_add_epi32(_mm256_set1_epi32(start_d1), _mm256_mullo_epi32(v_dE1_x, v_sequence));
        __m256i v_d2 = _mm256_add_epi32(_mm256_set1_epi32(start_d2), _mm256_mullo_epi32(v_dE2_x, v_sequence));
        __m256i v_d3 = _mm256_add_epi32(_mm256_set1_epi32(start_d3), _mm256_mullo_epi32(v_dE3_x, v_sequence));

        int x = x_min;

        for (; x < iters; x += 8) {
            __m256i mask_eq1  = _mm256_cmpeq_epi32(v_d1, v_zero); // ==
            __m256i mask_eq2  = _mm256_cmpeq_epi32(v_d2, v_zero);
            __m256i mask_eq3  = _mm256_cmpeq_epi32(v_d3, v_zero);

            __m256i mask_pos1 = _mm256_cmpgt_epi32(v_d1, v_zero); // >
            __m256i mask_ge1  = _mm256_or_si256(mask_pos1, mask_eq1); // or | ||

            __m256i mask_pos2 = _mm256_cmpgt_epi32(v_d2, v_zero);
            __m256i mask_ge2  = _mm256_or_si256(mask_pos2, mask_eq2);

            __m256i mask_pos3 = _mm256_cmpgt_epi32(v_d3, v_zero);
            __m256i mask_ge3  = _mm256_or_si256(mask_pos3, mask_eq3);

            __m256i mask_cw = _mm256_and_si256(_mm256_and_si256(mask_ge1, mask_ge2), mask_ge3);

            __m256i fr_b_v = _mm256_loadu_si256((__m256i*)&framebuffer[min(rows + rows_r + x + arr[0], len - 8)]);

            __m256i color_v = blend_pixels_avx(v_color, fr_b_v);

            _mm256_maskstore_epi32((int*)(&buffer[rows + x]), mask_cw, color_v);

            v_d1 = _mm256_add_epi32(v_d1, v_step1);
            v_d2 = _mm256_add_epi32(v_d2, v_step2);
            v_d3 = _mm256_add_epi32(v_d3, v_step3);
        }

        int d1 = start_d1 + (x - x_min) * dE1_x;
        int d2 = start_d2 + (x - x_min) * dE2_x;
        int d3 = start_d3 + (x - x_min) * dE3_x;

        for (; x < x_max; x++) {
            if ((d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0))
                buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);

            d1 += dE1_x;
            d2 += dE2_x;
            d3 += dE3_x;
        }

        start_d1 -= dE1_y;
        start_d2 -= dE2_y;
        start_d3 -= dE3_y;
        rows += width;
    }
}

/* триугольники конец */

static inline void draw_sprite(TGA_sprite *sprite, int width, int height, uint32_t *framebuffer){
    int start_x = max(sprite->loc.x, 0);
    int end_x = min(sprite->loc.x + sprite->width, width);
    int start_y = max(sprite->loc.y, 0);
    int end_y = min(sprite->loc.y + sprite->height, height);

    int bias_y = start_y == 0 ? 0 - sprite->loc.y : 0;
    int bias_x = start_x == 0 ? 0 - sprite->loc.x : 0;

    int rows = start_y * width;
    int len = width * height - 1;
    int rows_r = sprite->refraction[1] * width;

    #pragma omp parallel for schedule(static)
    for (int y = start_y; y < end_y; y++) {
        int rows_sprite = (y - start_y + bias_y) * sprite->width;

        for (int x = start_x; x < end_x; x++){
            uint32_t color = sprite->pixels[rows_sprite + (x - start_x + bias_x)];
            framebuffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + sprite->refraction[0], len)]);
        }
        rows += width;
    }
}

static inline void transparent_pixels(TGA_sprite *sprite, uint8_t r, uint8_t g, uint8_t b, float a){
    uint32_t *pixels = sprite->pixels;
    uint32_t a_u = a * 255.0f + 0.5f;
    a_u = a_u << 24;

    for (int i = 0; i < sprite->height * sprite->width; i++){
        uint32_t current_pixel = pixels[i];

        if (r == ((current_pixel >> 16) & 0xFF) && g == ((current_pixel >> 8)  & 0xFF) && b == (current_pixel & 0xFF)) pixels[i] = (current_pixel & 0x00FFFFFF) | a_u;
    }
}

static inline uint32_t alpha_writer(uint32_t color, float a){
    a = fminf(fmaxf(a, 0.0f), 1.0f);

    uint32_t a_u = a * 255.0f + 0.5f;
    return (color & 0x00FFFFFF) | (a_u << 24);
}

/* круги начало */
static inline void draw_circle(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_pixel(xc + x, yc + y, color, width, height, framebuffer); draw_pixel(xc - x, yc + y, color, width, height, framebuffer);
        draw_pixel(xc + x, yc - y, color, width, height, framebuffer); draw_pixel(xc - x, yc - y, color, width, height, framebuffer);
        draw_pixel(xc + y, yc + x, color, width, height, framebuffer); draw_pixel(xc - y, yc + x, color, width, height, framebuffer);
        draw_pixel(xc + y, yc - x, color, width, height, framebuffer); draw_pixel(xc - y, yc - x, color, width, height, framebuffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_circle_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_pixel_glass(xc + x, yc + y, color, width, height, framebuffer, fb); draw_pixel_glass(xc - x, yc + y, color, width, height, framebuffer, fb);
        draw_pixel_glass(xc + x, yc - y, color, width, height, framebuffer, fb); draw_pixel_glass(xc - x, yc - y, color, width, height, framebuffer, fb);
        draw_pixel_glass(xc + y, yc + x, color, width, height, framebuffer, fb); draw_pixel_glass(xc - y, yc + x, color, width, height, framebuffer, fb);
        draw_pixel_glass(xc + y, yc - x, color, width, height, framebuffer, fb); draw_pixel_glass(xc - y, yc - x, color, width, height, framebuffer, fb);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_circle_refraction(int xc, int yc, int r, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_pixel_refraction(xc + x, yc + y, color, arr, width, height, framebuffer, buffer); draw_pixel_refraction(xc - x, yc + y, color, arr, width, height, framebuffer, buffer);
        draw_pixel_refraction(xc + x, yc - y, color, arr, width, height, framebuffer, buffer); draw_pixel_refraction(xc - x, yc - y, color, arr, width, height, framebuffer, buffer);
        draw_pixel_refraction(xc + y, yc + x, color, arr, width, height, framebuffer, buffer); draw_pixel_refraction(xc - y, yc + x, color, arr, width, height, framebuffer, buffer);
        draw_pixel_refraction(xc + y, yc - x, color, arr, width, height, framebuffer, buffer); draw_pixel_refraction(xc - y, yc - x, color, arr, width, height, framebuffer, buffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line(xc - x, yc + y, xc + x, yc + y, color, width, height, framebuffer);
        draw_line(xc - x, yc - y, xc + x, yc - y, color, width, height, framebuffer);
        draw_line(xc - y, yc + x, xc + y, yc + x, color, width, height, framebuffer);
        draw_line(xc - y, yc - x, xc + y, yc - x, color, width, height, framebuffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line_glass(xc - x, yc + y, xc + x, yc + y, color, width, height, framebuffer, fb);
        draw_line_glass(xc - x, yc - y, xc + x, yc - y, color, width, height, framebuffer, fb);
        draw_line_glass(xc - y, yc + x, xc + y, yc + x, color, width, height, framebuffer, fb);
        draw_line_glass(xc - y, yc - x, xc + y, yc - x, color, width, height, framebuffer, fb);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle_refraction(int xc, int yc, int r, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line_refraction(xc - x, yc + y, xc + x, yc + y, color, arr, width, height, framebuffer, buffer);
        draw_line_refraction(xc - x, yc - y, xc + x, yc - y, color, arr, width, height, framebuffer, buffer);
        draw_line_refraction(xc - y, yc + x, xc + y, yc + x, color, arr, width, height, framebuffer, buffer);
        draw_line_refraction(xc - y, yc - x, xc + y, yc - x, color, arr, width, height, framebuffer, buffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle_avx(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line_avx(xc - x, xc + x, yc + y, color, width, height, framebuffer);
        draw_line_avx(xc - x, xc + x, yc - y, color, width, height, framebuffer);
        draw_line_avx(xc - y, xc + y, yc + x, color, width, height, framebuffer);
        draw_line_avx(xc - y, xc + y, yc - x, color, width, height, framebuffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle_avx_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line_avx_glass(xc - x, xc + x, yc + y, color, width, height, framebuffer, fb);
        draw_line_avx_glass(xc - x, xc + x, yc - y, color, width, height, framebuffer, fb);
        draw_line_avx_glass(xc - y, xc + y, yc + x, color, width, height, framebuffer, fb);
        draw_line_avx_glass(xc - y, xc + y, yc - x, color, width, height, framebuffer, fb);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

static inline void draw_filled_circle_avx_refraction(int xc, int yc, int r, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        draw_line_avx_refraction(xc - x, xc + x, yc + y, color, arr, width, height, framebuffer, buffer);
        draw_line_avx_refraction(xc - x, xc + x, yc - y, color, arr, width, height, framebuffer, buffer);
        draw_line_avx_refraction(xc - y, xc + y, yc + x, color, arr, width, height, framebuffer, buffer);
        draw_line_avx_refraction(xc - y, xc + y, yc - x, color, arr, width, height, framebuffer, buffer);

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}
/* круги конец */

/* квадраты начало */

static inline void draw_rect(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    int rows = start_y * width;

    //#pragma omp parallel for //collapse(2)
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            framebuffer[rows + x] = color;
        }
        rows += width;
    }
}

static inline void draw_rect_glass(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    int rows = start_y * width;

    //#pragma omp parallel for //collapse(2)
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            int index = rows + x;
            framebuffer[index] = blend_pixels(color, framebuffer[index]);
        }
        rows += width;
    }
}

static inline void draw_rect_refraction(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    int rows = start_y * width;
    int rows_r = arr[1] * width;
    int len = width * height - 1;

    //#pragma omp parallel for //collapse(2)
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);
        }
        rows += width;
    }
}

static inline void draw_rect_avx(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    __m256i vec_color = _mm256_set1_epi32(color);

    int rows = start_y * width;
    int iters = start_x + ((end_x - start_x) & ~7);

    for (int y = start_y; y < end_y; y++) {
        int x = start_x;

        for (; x < iters; x += 8) {
            _mm256_storeu_si256((__m256i*)&framebuffer[rows + x], vec_color);
        }

        for (; x < end_x; x++){
            framebuffer[rows + x] = color;
        }
        rows += width;
    }
}

static inline void draw_rect_avx_glass(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short width, short height, uint32_t *framebuffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    __m256i vec_color = _mm256_set1_epi32(color);

    int rows = start_y * width;
    int iters = start_x + ((end_x - start_x) & ~7);

    for (int y = start_y; y < end_y; y++) {
        int x = start_x;

        for (; x < iters; x += 8) {
            __m256i fr_b_v = _mm256_load_si256((__m256i*)&framebuffer[rows + x]);
            _mm256_storeu_si256((__m256i*)&framebuffer[rows + x], blend_pixels_avx(vec_color, fr_b_v));
        }

        for (; x < end_x; x++){
            int index = rows + x;
            framebuffer[index] = blend_pixels(color, framebuffer[index]);
        }

        rows += width;
    }
}

static inline void draw_rect_avx_refraction(int rect_x, int rect_y, int r_width, int r_height, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    int start_x = max(rect_x, 0);
    int end_x = min(rect_x + r_width, width);
    int start_y = max(rect_y, 0);
    int end_y = min(rect_y + r_height, height);

    __m256i vec_color = _mm256_set1_epi32(color);

    int rows = start_y * width;
    int iters = start_x + ((end_x - start_x) & ~7);
    int rows_r = arr[1] * width;
    int len = width * height - 1;

    for (int y = start_y; y < end_y; y++) {
        int x = start_x;

        for (; x < iters; x += 8) {
            __m256i fr_b_v = _mm256_loadu_si256((__m256i*)&framebuffer[min(rows + rows_r + x + arr[0], len - 8)]);
            _mm256_storeu_si256((__m256i*)&buffer[rows + x], blend_pixels_avx(vec_color, fr_b_v));
        }

        for (; x < end_x; x++){
            buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);
        }

        rows += width;
    }
}

static inline void init_rect(Rect *r){
    r->x[1] = r->x[0] + r->width;
    r->y[1] = r->y[0];

    r->x[2] = r->x[0] + r->width;
    r->y[2] = r->y[0] + r->height;

    r->x[3] = r->x[0];
    r->y[3] = r->y[0] + r->height;
}

/* квадраты конец */

static inline double get_time_in_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9L;
}

static inline void draw_char(int idx, int x, int y, unsigned short scale, uint32_t color, short width, short height, uint32_t *framebuffer) {
    const uint8_t *char_rows = font_data[idx];

    int current_y = y;

    for (int row = 0; row < 7; row++) {
        uint8_t bits = char_rows[row];
        if (bits == 0) {
            current_y += scale;
            continue;
        }

        draw_rect(x, current_y, scale * !!(bits & 0x10), scale , color, width, height, framebuffer);
        draw_rect(x + scale, current_y, scale * !!(bits & 0x08), scale , color, width, height, framebuffer);
        draw_rect(x + (scale * 2), current_y, scale * !!(bits & 0x04), scale, color, width, height, framebuffer);
        draw_rect(x + (scale * 3), current_y, scale * !!(bits & 0x02), scale, color, width, height, framebuffer);
        draw_rect(x + (scale * 4), current_y , scale * !!(bits & 0x01), scale , color, width, height, framebuffer);

        current_y += scale;
    }
}

static inline void draw_char_glass(int idx, int x, int y, unsigned short scale, uint32_t color, short width, short height, uint32_t *framebuffer) {
    const uint8_t *char_rows = font_data[idx];

    int current_y = y;

    for (int row = 0; row < 7; row++) {
        uint8_t bits = char_rows[row];
        if (bits == 0) {
            current_y += scale;
            continue;
        }

        draw_rect_glass(x, current_y, scale * !!(bits & 0x10), scale , color, width, height, framebuffer);
        draw_rect_glass(x + scale, current_y, scale * !!(bits & 0x08), scale , color, width, height, framebuffer);
        draw_rect_glass(x + (scale * 2), current_y, scale * !!(bits & 0x04), scale, color, width, height, framebuffer);
        draw_rect_glass(x + (scale * 3), current_y, scale * !!(bits & 0x02), scale, color, width, height, framebuffer);
        draw_rect_glass(x + (scale * 4), current_y , scale * !!(bits & 0x01), scale , color, width, height, framebuffer);

        current_y += scale;
    }
}

static inline void draw_char_refraction(int idx, int x, int y, unsigned short scale, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    const uint8_t *char_rows = font_data[idx];

    int current_y = y;

    for (int row = 0; row < 7; row++) {
        uint8_t bits = char_rows[row];
        if (bits == 0) {
            current_y += scale;
            continue;
        }

        draw_rect_refraction(x, current_y, scale * !!(bits & 0x10), scale , color, arr, width, height, framebuffer, buffer);
        draw_rect_refraction(x + scale, current_y, scale * !!(bits & 0x08), scale , color, arr, width, height, framebuffer, buffer);
        draw_rect_refraction(x + (scale * 2), current_y, scale * !!(bits & 0x04), scale, color, arr, width, height, framebuffer, buffer);
        draw_rect_refraction(x + (scale * 3), current_y, scale * !!(bits & 0x02), scale, color, arr, width, height, framebuffer, buffer);
        draw_rect_refraction(x + (scale * 4), current_y , scale * !!(bits & 0x01), scale , color, arr, width, height, framebuffer, buffer);

        current_y += scale;
    }
}

/* полигоны начало */

static inline void init_poligon(Vector2f* arr, int n, int x, int y, int r){
    float rad = 2.0f * M_PI * reciprocal(n);

    float dx, dy, cos_a, sin_a;

    arr[0].x = x;
    arr[0].y = y;

    arr[1].x = x;
    arr[1].y = y - r;

    cos_a = cosf(rad);
    sin_a = sinf(rad);

    for (int i = 1; i < n; i++){
        dx = arr[i].x - x;
        dy = arr[i].y - y;

        arr[i + 1].x = dx * cos_a - dy * sin_a + x;
        arr[i + 1].y = dx * sin_a + dy * cos_a + y;
    }
}

static inline void rotate_polygon(Vector2f *arr, int n, float rad){
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    for (int i = 1; i <= n; i++){
        float dx = arr[i].x - arr[0].x;
        float dy = arr[i].y - arr[0].y;

        arr[i].x = dx * cos_a - dy * sin_a + arr[0].x;
        arr[i].y = dx * sin_a + dy * cos_a + arr[0].y;
    }
}

static inline void draw_polygon_circuit(Vector2f *arr, int n, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (int i = 1; i < n; i++){
        draw_line(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, color, width, height, framebuffer);
    }
    draw_line(arr[n].x, arr[n].y, arr[1].x, arr[1].y, color, width, height, framebuffer);
}

static inline void draw_polygon(Vector2f *arr, int n, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (int i = 1; i < n; i++){
        draw_triangle(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, width, height, framebuffer);
    }
    draw_triangle(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color,  width, height, framebuffer);
}

static inline void draw_polygon_glass(Vector2f *arr, int n, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    for (int i = 1; i < n; i++){
        draw_triangle_glass(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, width, height, framebuffer, fb);
    }
    draw_triangle_glass(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color, width, height, framebuffer, fb);
}

static inline void draw_polygon_refraction(Vector2f *arr, int n, uint32_t color, short *arr_ref, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    for (int i = 1; i < n; i++){
        draw_triangle_refraction(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, arr_ref, width, height, framebuffer, buffer);
    }
    draw_triangle_refraction(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color, arr_ref,  width, height, framebuffer, buffer);
}

static inline void draw_polygon_avx(Vector2f *arr, int n, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (int i = 1; i < n; i++){
        draw_triangle_avx(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, width, height, framebuffer);
    }
    draw_triangle_avx(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color, width, height, framebuffer);
}

static inline void draw_polygon_avx_glass(Vector2f *arr, int n, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    for (int i = 1; i < n; i++){
        draw_triangle_avx_glass(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, width, height, framebuffer, fb);
    }
    draw_triangle_avx_glass(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color, width, height, framebuffer, fb);
}

static inline void draw_polygon_avx_refraction(Vector2f *arr, int n, uint32_t color, short *arr_ref, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    for (int i = 1; i < n; i++){
        draw_triangle_avx_refraction(arr[i].x, arr[i].y, arr[i + 1].x, arr[i + 1].y, arr[0].x, arr[0].y, color, arr_ref, width, height, framebuffer, buffer);
    }
    draw_triangle_avx_refraction(arr[n].x, arr[n].y, arr[1].x, arr[1].y, arr[0].x, arr[0].y, color, arr_ref, width, height, framebuffer, buffer);
}

/* полигоны конец */

/* строки начало */

static inline void draw_string(int x, int y, wchar_t *text, unsigned short scale, uint32_t color, short width, short height, uint32_t *framebuffer) {
    unsigned short i = 0;
    while (text[i] != '\0') {
        draw_char(get_char_index(text[i]), x + i * 6 * scale, y, scale, color,  width, height, framebuffer);
        i++;
    }
}

static inline void draw_string_glass(int x, int y, wchar_t *text, unsigned short scale, uint32_t color, short width, short height, uint32_t *framebuffer) {
    unsigned short i = 0;
    while (text[i] != '\0') {
        draw_char_glass(get_char_index(text[i]), x + i * 6 * scale, y, scale, color,  width, height, framebuffer);
        i++;
    }
}

static inline void draw_string_refraction(int x, int y, wchar_t *text, unsigned short scale, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    unsigned short i = 0;
    while (text[i] != '\0') {
        draw_char_refraction(get_char_index(text[i]), x + i * 6 * scale, y, scale, color, arr, width, height, framebuffer, buffer);
        i++;
    }
}

static inline void draw_len_string(int x, int y, wchar_t *text, unsigned short scale, unsigned int len, uint32_t color, short width, short height, uint32_t* framebuffer) {
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < len; i++) {
        draw_char(get_char_index(text[i]), x + i * 6 * scale, y, scale, color, width, height, framebuffer);
    }
}

static inline void draw_len_string_glass(int x, int y, wchar_t *text, unsigned short scale, unsigned int len, uint32_t color, short width, short height, uint32_t *framebuffer) {
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < len; i++) {
        draw_char_glass(get_char_index(text[i]), x + i * 6 * scale, y, scale, color,  width, height, framebuffer);
    }
}

static inline void draw_len_string_refraction(int x, int y, wchar_t *text, unsigned short scale, unsigned int len, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < len; i++) {
        draw_char_refraction(get_char_index(text[i]), x + i * 6 * scale, y, scale, color, arr, width, height, framebuffer, buffer);
    }
}

/* строки конец */

static inline int init_elips(int a, int b, int x_c, int y_c, Vector2f *arr) {
    int max_q_size = a + b + 2;

    Vector2i *q = (Vector2i*)__builtin_alloca(max_q_size * sizeof(Vector2i));

    int x = 0;
    int y = b;

    long a_2 = a * a;
    long b_2 = b * b;

    long d1 = b_2 - (a_2 * b) + (a_2 >> 2);

    long dx = (b_2 * x) << 1;
    long dy = (a_2 * y) << 1;

    int q_count = 0;

    for (; dx < dy; q_count++) {
        q[q_count] = (Vector2i){x, y};

        if (d1 < 0){
            x++;
            dx += b_2 << 1;
            d1 += dx + b_2;
        } else {
            x++;
            y--;
            dx += b_2 << 1;
            dy -= a_2 << 1;
            d1 += dx - dy + b_2;
        }
    }

    long d2 = (b_2 * ((x + 0.5) * (x + 0.5))) + (a_2 * ((y - 1) * (y - 1))) - (a_2 * b_2);

    for (; y >= 0; q_count++){
        q[q_count] = (Vector2i){x, y};

        if (d2 > 0){
            y--;
            dy -= a_2 << 1;
            d2 += a_2 - dy;
        } else {
            x++;
            y--;
            dx += b_2 << 1;
            dy -= a_2 << 1;
            d2 += dx - dy + a_2;
        }
    }

    int idx = 1;

    arr[0] = (Vector2f){(float)x_c, (float)y_c};

    for (int i = 0; i < q_count; i++) {
        arr[idx].x = (x_c + q[i].x);
        arr[idx].y = (y_c + q[i].y);
        idx++;
    }

    for (int i = q_count - 2; i >= 0; i--) {
        arr[idx].x = (x_c + q[i].x);
        arr[idx].y = (y_c - q[i].y);
        idx++;
    }

    for (int i = 1; i < q_count; i++) {
        arr[idx].x = (x_c - q[i].x);
        arr[idx].y = (y_c - q[i].y);
        idx++;
    }

    for (int i = q_count - 2; i > 0; i--) {
        arr[idx].x = (x_c - q[i].x);
        arr[idx].y = (y_c + q[i].y);
        idx++;
    }

    return idx - 1;
}

static inline bool is_circle_colission(Vector2f *loc_1, Vector2f *loc_2, float r1, float r2){
    float x = (loc_2->x - loc_1->x);
    float y = (loc_2->y - loc_1->y);
    float lenght = sqrtf(x * x + y * y) + FLT_MIN;
    float len = (r1 + r2 - lenght);

    if (len >= 0.0f) return true;
    return false;
}

static inline void circle_colission(Vector2f *loc_1, Vector2f *loc_2, float r1, float r2, float mass1, float mass2){
    float x = (loc_2->x - loc_1->x);
    float y = (loc_2->y - loc_1->y);
    float lenght = sqrtf(x * x + y * y) + FLT_MIN;
    float len = (r1 + r2 - lenght);

    float factor = len * reciprocal(lenght) * reciprocal(mass1 + mass2 + FLT_MIN);

    loc_1->x -= x * factor * mass2;
    loc_1->y -= y * factor * mass2;

    loc_2->x += x * factor * mass1;
    loc_2->y += y * factor * mass1;
}

static inline void acceleration(Vector2f *speed, const float mass, const Vector2f *max_speed, const Vector2f *f, const Vector2s *move, bool use_g, const double delta_time){
    speed->x += f->x * move->x * reciprocal(mass) * delta_time;
    speed->y += f->y * move->y * reciprocal(mass) * delta_time;

    if (use_g){
        if (speed->x > 0.0f){
            speed->x -= M_G * GROUND_MU_TREN;
            speed->x = fmaxf(0.0f, speed->x);
        } else if (speed->x < 0.0f){
            speed->x += M_G * GROUND_MU_TREN;
            speed->x = fminf(0.0f, speed->x);
        }

        if (speed->y > 0.0f){
            speed->y -= M_G * GROUND_MU_TREN;
            speed->y = fmaxf(0.0f, speed->y);
        } else if (speed->y < 0.0f){
            speed->y += M_G * GROUND_MU_TREN;
            speed->y = fminf(0.0f, speed->y);
        }
    }

    speed->x = fminf(fabsf(speed->x), max_speed->x) * SIGN(speed->x);
    speed->y = fminf(fabsf(speed->y), max_speed->y) * SIGN(speed->y);
}

static inline Vector2f force_graviti(const Vector2f *loc_1, const Vector2f *loc_2, const float mass_1, const float mass_2){
    float x = (loc_2->x - loc_1->x);
    float y = (loc_2->y - loc_1->y);
    float lenght = x * x + y * y + FLT_MIN;
    float f = M_GRAV * mass_1 * mass_2 * reciprocal(lenght);

    lenght = reciprocal(sqrtf(lenght));

    Vector2f f_v = {x * lenght * f, y * lenght * f};

    return f_v;
}

static inline Vector2f force_kulon(const Vector2f *loc_1, const Vector2f *loc_2, const float q_1, const float q_2, const float e){
    float x = (loc_2->x - loc_1->x);
    float y = (loc_2->y - loc_1->y);
    float lenght = x * x + y * y + FLT_MIN;
    float f = (M_K * reciprocal(e)) * (fabsf(q_1) + FLT_MIN) * (fabsf(q_2) + FLT_MIN) * reciprocal(lenght) * SIGN(q_1) * SIGN(q_2) * (-1.0f);

    lenght = reciprocal(sqrtf(lenght));

    Vector2f f_v = {x * lenght * f, y * lenght * f};

    return f_v;
}

static inline Vector2f force_magnitizm(const Vector2f *loc_1, const Vector2f *loc_2, const float maga_1, const float maga_2){
    float x = (loc_2->x - loc_1->x);
    float y = (loc_2->y - loc_1->y);
    float lenght = x * x + y * y + FLT_MIN;
    float f = (3.0f * M_MAGNITIC * reciprocal(2.0f * M_PI) * (maga_1 * maga_2 * reciprocal(lenght * lenght)));

    lenght = reciprocal(sqrtf(lenght));

    Vector2f f_v = {x * lenght * f, y * lenght * f};

    return f_v;
}

static inline void Vector2f_add_(Vector2f *a, Vector2f *b){
    a->x += b->x;
    a->y += b->y;
}

static inline void Vector2s_add_(Vector2s *a, Vector2s *b){
    a->x += b->x;
    a->y += b->y;
}

// 0.0f <= elasticity <= 1.0f
// static inline void elasticity(Vector2f v_1, Vector2f v_2, float el_1, float el_2){
//     float e = (el_1 + el_2) * 0.5f;

// }

static inline Vector2f screen_mouse_loc_to_map(Vector2s *mouse_loc, float SCREEN_WIDTH, float SCREEN_HEIGHT, Camera *camera){
    float m_dx = mouse_loc->x - SCREEN_WIDTH * 0.5f;
    float m_dy = mouse_loc->y - SCREEN_HEIGHT * 0.5f;

    float rad = -(M_PI * reciprocal(180.0f)) * camera->rad;
    float cos_cam = cosf(rad);
    float sin_cam = sinf(rad);

    Vector2f mouse_delt = {m_dx * cos_cam - m_dy * sin_cam, m_dx * sin_cam + m_dy * cos_cam};

    return (Vector2f){(mouse_delt.x - camera->loc.x), (mouse_delt.y - camera->loc.y)};
}

static inline void box_blur_2d_uint32_fast(uint32_t* buffer, int width, int height, int radius) {
    if (radius <= 0) return;

    int window_size = radius * 2 + 1;

    uint64_t scale = ((1ULL << 32) + window_size - 1) / window_size;

    uint32_t* temp_buffer = (uint32_t*)malloc(width * height * sizeof(uint32_t));

    for (int y = 0; y < height; y++) {
        int row_offset = y * width;
        uint32_t sum_a = 0, sum_r = 0, sum_g = 0, sum_b = 0;

        for (int i = -radius; i <= radius; i++) {
            int sample_x = (i < 0) ? 0 : ((i >= width) ? width - 1 : i);
            uint32_t pixel = buffer[row_offset + sample_x];
            sum_a += (pixel >> 24) & 0xFF;
            sum_r += (pixel >> 16) & 0xFF;
            sum_g += (pixel >> 8) & 0xFF;
            sum_b += pixel & 0xFF;
        }

        for (int x = 0; x < width - 1; x++) {
            uint32_t a = (sum_a * scale) >> 32;
            uint32_t r = (sum_r * scale) >> 32;
            uint32_t g = (sum_g * scale) >> 32;
            uint32_t b = (sum_b * scale) >> 32;

            temp_buffer[row_offset + x] = (a << 24) | (r << 16) | (g << 8) | b;

            int next_x = min(x + radius + 1, width - 1);
            int prev_x = max(x - radius, 0);

            uint32_t next_pixel = buffer[row_offset + next_x];
            uint32_t prev_pixel = buffer[row_offset + prev_x];

            sum_a += ((next_pixel >> 24) & 0xFF) - ((prev_pixel >> 24) & 0xFF);
            sum_r += ((next_pixel >> 16) & 0xFF) - ((prev_pixel >> 16) & 0xFF);
            sum_g += ((next_pixel >> 8) & 0xFF) - ((prev_pixel >> 8) & 0xFF);
            sum_b += (next_pixel & 0xFF) - (prev_pixel & 0xFF);
        }
    }

    uint32_t* col_sum_a = (uint32_t*)calloc(width, sizeof(uint32_t));
    uint32_t* col_sum_r = (uint32_t*)calloc(width, sizeof(uint32_t));
    uint32_t* col_sum_g = (uint32_t*)calloc(width, sizeof(uint32_t));
    uint32_t* col_sum_b = (uint32_t*)calloc(width, sizeof(uint32_t));

    for (int i = -radius; i <= radius; i++) {
        int sample_y = (i < 0) ? 0 : ((i >= height) ? height - 1 : i);
        for (int x = 0; x < width; x++) {
            uint32_t pixel = temp_buffer[sample_y * width + x];
            col_sum_a[x] += (pixel >> 24) & 0xFF;
            col_sum_r[x] += (pixel >> 16) & 0xFF;
            col_sum_g[x] += (pixel >> 8) & 0xFF;
            col_sum_b[x] += pixel & 0xFF;
        }
    }

    for (int y = 0; y < height; y++) {
        int row_offset = y * width;

        for (int x = 0; x < width; x++) {
            uint32_t a = (col_sum_a[x] * scale) >> 32;
            uint32_t r = (col_sum_r[x] * scale) >> 32;
            uint32_t g = (col_sum_g[x] * scale) >> 32;
            uint32_t b = (col_sum_b[x] * scale) >> 32;

            buffer[row_offset + x] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        if (y < height - 1) {
            int next_y = min(height - 1, y + radius + 1);
            int prev_y = max(y - radius, 0);

            int next_row = next_y * width;
            int prev_row = prev_row = prev_y * width;

            for (int x = 0; x < width; x++) {
                uint32_t next_pixel = temp_buffer[next_row + x];
                uint32_t prev_pixel = temp_buffer[prev_row + x];

                col_sum_a[x] += ((next_pixel >> 24) & 0xFF) - ((prev_pixel >> 24) & 0xFF);
                col_sum_r[x] += ((next_pixel >> 16) & 0xFF) - ((prev_pixel >> 16) & 0xFF);
                col_sum_g[x] += ((next_pixel >> 8) & 0xFF) - ((prev_pixel >> 8) & 0xFF);
                col_sum_b[x] += (next_pixel & 0xFF) - (prev_pixel & 0xFF);
            }
        }
    }

    free(col_sum_a); free(col_sum_r); free(col_sum_g); free(col_sum_b);
    free(temp_buffer);
}

#endif // CORE_H
