#ifndef CORE_H
#define CORE_H

#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "font.h"

#define FLT_MAX  3.402823e+38f
#define FLT_MIN 0.000000000000000000000000000000000001f

typedef struct{
    float speed;
    float scale;
    float x[4];
    float y[4];
    float width;
    float height;
    uint32_t color;
    float deg;
    short refraction[2];
} Rect;

typedef struct{
    float speed;
    unsigned short scale;
    float x;
    float y;
    wchar_t text[32];
    uint32_t color;
    unsigned int len;
    short refraction[2];
} Text;

typedef struct {
    float x[2];
    float y[2];
    float speed;
    uint32_t color;
    float deg;
    short refraction[2];
} Line;

typedef struct {
    float x;
    float y;
    float speed;
    uint32_t color;
    short refraction[2];
} Pixel;

typedef struct {
    uint16_t width;
    uint16_t height;
    uint32_t *pixels;
    float x;
    float y;
    float speed;
    short refraction[2];
} TGA_sprite;

typedef struct{
    float speed;
    float scale;
    float x;
    float y;
    float r;
    uint32_t color;
    short refraction[2];
    float mass;
} Circle;

typedef struct{
    float speed;
    float scale;
    float x[3];
    float y[3];
    uint32_t color;
    float deg;
    short refraction[2];
} Triangle;

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
    int start_x = max(sprite->x, 0);
    int end_x = min(sprite->x + sprite->width, width);
    int start_y = max(sprite->y, 0);
    int end_y = min(sprite->y + sprite->height, height);

    int bias_y = start_y == 0 ? 0 - sprite->y : 0;
    int bias_x = start_x == 0 ? 0 - sprite->x : 0;

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

static inline void init_poligon(float* arr_x, float* arr_y, short n, int x, int y, short r){
    float rad = 2.0f * M_PI * reciprocal(n);

    float dx, dy, cos_a, sin_a;

    arr_x[0] = x;
    arr_y[0] = y - r;

    cos_a = cosf(rad);
    sin_a = sinf(rad);

    for (short i = 0; i < n - 1; i++){
        dx = arr_x[i] - x;
        dy = arr_y[i] - y;

        arr_x[i + 1] = dx * cos_a - dy * sin_a + x;
        arr_y[i + 1] = dx * sin_a + dy * cos_a + y;
    }
}

static inline void rotate_polygon(float *x, float *y, short n, float rad){
    float cx = (max_vector(x, n) + min_vector(x, n)) * 0.5f;
    float cy = (max_vector(y, n) + min_vector(y, n)) * 0.5f;

    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    float dx, dy;

    for (short i = 0; i < n; i++){
        dx = x[i] - cx;
        dy = y[i] - cy;

        x[i] = dx * cos_a - dy * sin_a + cx;
        y[i] = dx * sin_a + dy * cos_a + cy;
    }
}

static inline void draw_polygon_circuit(float *x, float *y, short n, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_line(x[i], y[i], x[0], y[0], color, width, height, framebuffer);
            return;
        }
        draw_line(x[i], y[i], x[i + 1], y[i + 1], color, width, height, framebuffer);
    }
}

static inline void draw_polygon(float *x, float *y, short n, int xc, int yc, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle(x[i], y[i], x[0], y[0], xc, yc, color,  width, height, framebuffer);
            return;
        }
        draw_triangle(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, width, height, framebuffer);
    }
}

static inline void draw_polygon_glass(float *x, float *y, short n, int xc, int yc, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle_glass(x[i], y[i], x[0], y[0], xc, yc, color, width, height, framebuffer, fb);
            return;
        }
        draw_triangle_glass(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, width, height, framebuffer, fb);
    }
}

static inline void draw_polygon_refraction(float *x, float *y, short n, int xc, int yc, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle_refraction(x[i], y[i], x[0], y[0], xc, yc, color, arr,  width, height, framebuffer, buffer);
            return;
        }
        draw_triangle_refraction(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, arr, width, height, framebuffer, buffer);
    }
}

static inline void draw_polygon_avx(float *x, float *y, short n, int xc, int yc, uint32_t color, short width, short height, uint32_t *framebuffer){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle_avx(x[i], y[i], x[0], y[0], xc, yc, color, width, height, framebuffer);
            return;
        }
        draw_triangle_avx(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, width, height, framebuffer);
    }
}

static inline void draw_polygon_avx_glass(float *x, float *y, short n, int xc, int yc, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle_avx_glass(x[i], y[i], x[0], y[0], xc, yc, color, width, height, framebuffer, fb);
            return;
        }
        draw_triangle_avx_glass(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, width, height, framebuffer, fb);
    }
}

static inline void draw_polygon_avx_refraction(float *x, float *y, short n, int xc, int yc, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    for (short i = 0; i < n; i++){
        if (i == (n - 1)){
            draw_triangle_avx_refraction(x[i], y[i], x[0], y[0], xc, yc, color, arr, width, height, framebuffer, buffer);
            return;
        }
        draw_triangle_avx_refraction(x[i], y[i], x[i + 1], y[i + 1], xc, yc, color, arr, width, height, framebuffer, buffer);
    }
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

static inline void project_polygon(float *poly_x, float *poly_y, int count, float axis_x, float axis_y, float *min, float *max) {
    *min = FLT_MAX;
    *max = -FLT_MAX;
    for (short i = 0; i < count; i++) {
        float dot = (poly_x[i] * axis_x) + (poly_y[i] * axis_y);
        *min = fminf(dot, *min);
        *max = fmaxf(dot, *max);
    }
}

static inline bool is_colission(float *x1, float *y1, float *x2, float *y2, float *d_x, float *d_y) {
    *d_x = 0.0f;
    *d_y = 0.0f;

    float overlap = FLT_MAX;
    float smallest_axis_x = 0.0f;
    float smallest_axis_y = 0.0f;

    for (short i = 0; i < 4; i++) {
        short next = (i + 1) & ~3;

        float edge_x = x1[next] - x1[i];
        float edge_y = y1[next] - y1[i];

        float axis_x = -edge_y;
        float axis_y = edge_x;

        float length = sqrtf(axis_x * axis_x + axis_y * axis_y);
        if (length == 0.0f) continue;
        axis_x /= length;
        axis_y /= length;

        float min1, max1, min2, max2;
        project_polygon(x1, y1, 4, axis_x, axis_y, &min1, &max1);
        project_polygon(x2, y2, 4, axis_x, axis_y, &min2, &max2);

        if (max1 < min2 || max2 < min1) return false;

        float current_overlap = fminf(max1, max2) - fmaxf(min1, min2);

        if (current_overlap < overlap) {
            overlap = current_overlap;
            smallest_axis_x = axis_x;
            smallest_axis_y = axis_y;
        }
    }

    for (short i = 0; i < 4; i++) {
        short next = (i + 1) & ~3;

        float edge_x = x2[next] - x2[i];
        float edge_y = y2[next] - y2[i];

        float axis_x = -edge_y;
        float axis_y = edge_x;

        float length = sqrtf(axis_x * axis_x + axis_y * axis_y);
        if (length == 0.0f) continue;
        axis_x /= length;
        axis_y /= length;

        float min1, max1, min2, max2;
        project_polygon(x1, y1, 4, axis_x, axis_y, &min1, &max1);
        project_polygon(x2, y2, 4, axis_x, axis_y, &min2, &max2);

        if (max1 < min2 || max2 < min1) return false;

        float current_overlap = fminf(max1, max2) - fmaxf(min1, min2);

        if (current_overlap < overlap) {
            overlap = current_overlap;
            smallest_axis_x = axis_x;
            smallest_axis_y = axis_y;
        }
    }

    float c1_x = (x1[0] + x1[2]) * 0.5f;
    float c1_y = (y1[0] + y1[2]) * 0.5f;
    float c2_x = (x2[0] + x2[2]) * 0.5f;
    float c2_y = (y2[0] + y2[2]) * 0.5f;

    float d_centers_x = c1_x - c2_x;
    float d_centers_y = c1_y - c2_y;
    float dot_direction = (d_centers_x * smallest_axis_x) + (d_centers_y * smallest_axis_y);

    dot_direction = (fabsf(dot_direction) * reciprocal(dot_direction));

    smallest_axis_x = smallest_axis_x * dot_direction;
    smallest_axis_y = smallest_axis_y * dot_direction;

    *d_x = smallest_axis_x * overlap;
    *d_y = smallest_axis_y * overlap;

    return true;
}

static inline void is_circle_colission(float *x1, float *y1, float *x2, float *y2, float r1, float r2, float mass1, float mass2){
    float x = (*x2 - *x1);
    float y = (*y2 - *y1);
    float lenght = sqrtf(x * x + y * y) + FLT_MIN;
    float len = (r1 + r2 - lenght);

    if (len >= 0.0f){
        float factor = len * reciprocal(lenght) * reciprocal(mass1 + mass2 + FLT_MIN);

        *x1 -= x * factor * mass2;
        *y1 -= y * factor * mass2;

        *x2 += x * factor * mass1;
        *y2 += y * factor * mass1;
    }
}

#endif // CORE_H
