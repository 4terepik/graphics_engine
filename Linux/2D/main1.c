#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <X11/Xutil.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h> 
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <alsa/asoundlib.h>
#include "font.h"

//sudo apt install libx11-dev && sudo apt install libxext-dev && sudo apt install libasound2-dev && sudo apt install gcc-multilib g++-multilib - Debian
//sudo dnf install libX11-devel && sudo dnf install libXext-devel && sudo dnf install glibc-devel.i686 libstdc++-devel.i686 - Fedora/RHEL
//sudo pacman -S libx11 && sudo pacman -S libxext && sudo pacman -Syu gcc-multilib - Arch

//gcc Linux/2D/main.c -lX11 -Wall -Wextra -Werror -fopenmp -lXext -Ofast -ffast-math -lm -lasound -s -mavx2
// -lm <math.h>
// -O0 без оптимезации
// -O1 чуть оптимизирован
// -O2 норм оптимезаци
// -O3 сильная оптимецаи
// -Ofast очень сильная оптимезация, может быть неточность при подсчете с float и double
// -g откладка
// -mavx2 для AVX интсрукций
// -march=native для AVX интсрукций но под конкретное железо
// -m32 для 32 битных процов

#define SAMPLE_RATE 44100
#define LATENT 50000

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

static inline bool save_tga(const char *filename, short width, short height, uint32_t *framebuffer){

    FILE *f = fopen(filename, "wb");

    if (!f) {
        fprintf(stderr, "Ошибка: Не удалось создать файл %s\n", filename);
        return false;
    }
    
    TGAHeader header = {0};
    header.image_type = 2;
    header.width = (uint16_t)width;
    header.height = (uint16_t)height;
    header.bits_per_pixel = 32;
    header.image_descriptor = 0x20;

    uint32_t *buffer = (uint32_t *)malloc(sizeof(uint32_t) * width * height);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; i++) {
        buffer[i] = framebuffer[i] | 0xFF000000;
    }

    fwrite(&header, sizeof(TGAHeader), 1, f);
    fwrite(buffer, sizeof(uint32_t), width * height, f);
    fflush(f);
    fclose(f);
    free(buffer);

    return true;
}

static inline bool load_tga(const char *filename, TGA_sprite *out_texture) {
    FILE *file = fopen(filename, "rb");

    if (!file) {
        fprintf(stderr, "Ошибка: Не удалось открыть файл %s\n", filename);
        return 0;
    }

    TGAHeader header;
    if (fread(&header, sizeof(TGAHeader), 1, file) != 1) {
        fprintf(stderr, "Ошибка чтения заголовка TGA: %s\n", filename);
        fclose(file);
        return false;
    }

    if (header.image_type != 2) {
        fprintf(stderr, "Ошибка: Движок поддерживает только несжатый TGA (тип 2). Файл: %s\n", filename);
        fclose(file);
        return false;
    }

    if (header.bits_per_pixel != 32) {
        fprintf(stderr, "Ошибка: Движок требует строго 32-битный TGA с альфа-каналом. Файл: %s\n", filename);
        fclose(file);
        return false;
    }

    if (header.id_length > 0) fseek(file, header.id_length, SEEK_CUR);

    uint32_t pixel_count = header.width * header.height;
    out_texture->pixels = (uint32_t *)malloc(pixel_count * sizeof(uint32_t));

    if (fread(out_texture->pixels, sizeof(uint32_t), pixel_count, file) != pixel_count) {
        fprintf(stderr, "Ошибка чтения пикселей TGA: %s\n", filename);
        free(out_texture->pixels);
        fclose(file);
        return false;
    }

    out_texture->width = header.width;
    out_texture->height = header.height;

    fclose(file);
    return true;
}

static inline short *load_wav(const char *filename, WavHeader *header) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", filename);
        return NULL;
    }

    if (fread(header, sizeof(WavHeader), 1, file) != 1) {
        fprintf(stderr, "Ошибка чтения заголовка\n");
        fclose(file);
        return NULL;
    }

    if (strncmp(header->chunk_id, "RIFF", 4) != 0 || strncmp(header->format, "WAVE", 4) != 0) {
        fprintf(stderr, "Ошибка: файл не является форматом WAV\n");
        fclose(file);
        return NULL;
    }

    short* buffer = (short*)malloc(header->subchunk2_size);

    if (fread(buffer, 1, header->subchunk2_size, file) == 0) 
        printf("Предупреждение: аудиоданные не считались или файл пустой\n");

    fclose(file);
    
    return buffer;
}

static inline void play_wav(short *buffer, WavHeader *header, snd_pcm_t *handle) {
    snd_pcm_format_t format = (header->bits_per_sample == 16) ? SND_PCM_FORMAT_S16_LE : SND_PCM_FORMAT_U8;
    
    snd_pcm_set_params(handle, format, SND_PCM_ACCESS_RW_INTERLEAVED, header->num_channels, header->sample_rate, 1, LATENT);

    uint32_t total_frames = header->subchunk2_size / (header->num_channels * (header->bits_per_sample / 8));

    #pragma omp task firstprivate(buffer, total_frames, handle)
    {
        snd_pcm_sframes_t frames = snd_pcm_writei(handle, buffer, total_frames);
        if (frames < 0) {
            snd_pcm_prepare(handle);
        }

        snd_pcm_drain(handle); 
    }
}

static inline void apply_volume(short *buffer, uint32_t bytes_count, float volume) {
    uint32_t samples_count = bytes_count / sizeof(short);

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < samples_count; i++) {
        buffer[i] = (short)max(-32767, min((buffer[i] * volume), 32767));
    }
}

static inline void sin_play(snd_pcm_t *handle, float time, double frequency, double volume) {
    short *buffer = (short*)malloc((size_t)(SAMPLE_RATE * 2 * time * sizeof(short)));
    // short *buffer = (short*)calloc((size_t)(SAMPLE_RATE * 2 * time), sizeof(short));

    snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 2, SAMPLE_RATE, 1, LATENT);

    #pragma omp parallel for
    for (int i = 0; i < (int)(SAMPLE_RATE * time); i++) {
        short sample = 32767.0 * volume * sin(2.0 * M_PI * frequency * i / SAMPLE_RATE);
        buffer[i * 2] = sample;
        buffer[i * 2 + 1] = sample;
    }

    #pragma omp task firstprivate(buffer, handle, time)
    {
        snd_pcm_sframes_t frames = snd_pcm_writei(handle, buffer, (int)(SAMPLE_RATE * time));
        if (frames < 0) {
            snd_pcm_prepare(handle);
        }

        snd_pcm_drain(handle); 

        free(buffer);
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

// ⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⣀⣀⠀⠀⠀⠀⠀⠀
// ⠀⠀⠀⠀⢀⣴⣾⣿⣿⣿⣶⣄⠈⠑⢀⠀⠀⠀
// ⠀⠀⣠⣾⣿⣿⣿⣿⣛⡛⠛⠛⠷⣶⣤⡆⠀⠀
// ⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣦⣀⣀⡀⢿⠇⠀⠀
// ⠐⣿⣿⣿⣿⣿⣿⠋⠀⣠⣄⡀⠈⠙⣾⣤⠀⠀
// ⠀⠈⢛⡿⢿⣿⠆⠀⠘⠛⢻⠿⠁⢿⡟⠉⠀⠀
// ⠀⠀⠀⣿⡟⠀⠀⠀⠀⠀⢀⠆⣤⡄⠁⠀⠀⠀
// ⠀⠀⠀⢹⠢⠄⠀⠀⠀⢀⣼⡿⠿⣷⠀⠀⠀⠀
// ⠀⠀⢀⡈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠃⠀⠀⠀
// ⠀⢀⣎⣸⣿⣶⣤⣀⠀⠐⠶⢖⠖⠁⠀⠀⠀⠀
// ⠐⠠⢝⡛⠿⠛⠛⠛⠓⠢⢠⠏⢆⠀⠀⠀⠀⠀
// ⠀⠀⠀⠈⠓⢤⡀⠀⠀⢀⠊⢦⡈⠢⡀⠀⠀⠀
// ⠀⠀⠀⠀⠀⠀⠉⠳⠤⠌⠀⠀⠙⠢⣄⠑⡀⠀
// ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠊⠐

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

static inline void draw_filled_circle_1(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer){
    int y_min = max(yc - r, 0);
    int y_max = (yc + r >= height) ? height - 1 : yc + r;
    int x_min = max(xc - r, 0);
    int x_max = (xc + r >= width) ? width - 1 : xc + r;

    r *= r;

    int rows = y_min * width, dy = 0, dx = 0;

    for (int y = y_min; y <= y_max; y++){
        dy = y - yc;
        dy *= dy;

        for (int x = x_min; x <= x_max; x++){
            dx = x - xc;
            dx *= dx;

            if (dx + dy <= r) framebuffer[rows + x] = color;
        }
        rows += width;
    }
}

static inline void draw_filled_circle_1_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb){
    int y_min = max(yc - r, 0);
    int y_max = (yc + r >= height) ? height - 1 : yc + r;
    int x_min = max(xc - r, 0);
    int x_max = (xc + r >= width) ? width - 1 : xc + r;

    r *= r;

    int rows = y_min * width, dy = 0, dx = 0;

    for (int y = y_min; y <= y_max; y++){
        dy = y - yc;
        dy *= dy;

        for (int x = x_min; x <= x_max; x++){
            dx = x - xc;
            dx *= dx;

            int index = rows + x;

            if (dx + dy <= r && fb[index] == 0) {
                framebuffer[index] = blend_pixels(color, framebuffer[index]);
                fb[index] = 1;
            }
        }
        rows += width;
    }
}

static inline void draw_filled_circle_1_refraction(int xc, int yc, int r, uint32_t color, int *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer){
    int y_min = max(yc - r, 0);
    int y_max = (yc + r >= height) ? height - 1 : yc + r;
    int x_min = max(xc - r, 0);
    int x_max = (xc + r >= width) ? width - 1 : xc + r;

    r *= r;

    int rows = y_min * width, dy = 0, dx = 0, rows_r = arr[1] * width, len = width * height - 1;

    for (int y = y_min; y <= y_max; y++){
        dy = y - yc;
        dy *= dy;

        for (int x = x_min; x <= x_max; x++){
            dx = x - xc;
            dx *= dx;

            if (dx + dy <= r) buffer[rows + x] = blend_pixels(color, framebuffer[min(rows + rows_r + x + arr[0], len)]);
        }
        rows += width;
    }
}


static inline void draw_filled_circle_2(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer) {
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

static inline void draw_filled_circle_2_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
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

static inline void draw_filled_circle_2_refraction(int xc, int yc, int r, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
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

static inline void draw_filled_circle_2_avx(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer) {
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

static inline void draw_filled_circle_2_avx_glass(int xc, int yc, int r, uint32_t color, short width, short height, uint32_t *framebuffer, uint8_t *fb) {
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

static inline void draw_filled_circle_2_avx_refraction(int xc, int yc, int r, uint32_t color, short *arr, short width, short height, uint32_t *framebuffer, uint32_t *buffer) {
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

static inline bool is_in_box(float *x_arr, float *y_arr, float x, float y){
    int d1, d2, d3, d4;

    d1 = (x_arr[1] - x_arr[0]) * (y - y_arr[0]) - (y_arr[1] - y_arr[0]) * (x - x_arr[0]);
    d2 = (x_arr[2] - x_arr[1]) * (y - y_arr[1]) - (y_arr[2] - y_arr[1]) * (x - x_arr[1]);
    d3 = (x_arr[3] - x_arr[2]) * (y - y_arr[2]) - (y_arr[3] - y_arr[2]) * (x - x_arr[2]);
    d4 = (x_arr[0] - x_arr[3]) * (y - y_arr[3]) - (y_arr[0] - y_arr[3]) * (x - x_arr[3]);

    return (d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0) || (d1 < 0 && d2 < 0 && d3 < 0 && d4 < 0);
}

int main(void) {
    short SCREEN_WIDTH = 1300;
    short SCREEN_HEIGHT = 900;

    uint32_t *framebuffer;

    uint32_t *framebuffer_2 = (uint32_t*)aligned_alloc(32, (SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint32_t) + 31) & ~31);

    /* для широких строк */
    setlocale(LC_ALL, "");

    /* инициализация звуковой карты */
    snd_pcm_t *handle;

    if (snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
        fprintf(stderr, "Ошибка аудио! Игра запустится без звука.\n");
    }

    
    /* инициализация шрифта */
    init_char_lut();

    /* инициализация окна х11 */
    bool keys[65536] = {false};
    bool buttons[8] = {false}; 

    Display *display = XOpenDisplay(NULL);

    int screen = DefaultScreen(display);
    Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 1, BlackPixel(display, screen), WhitePixel(display, screen));

    /* фиксируем размер окна*/

    // XSizeHints *hints = XAllocSizeHints();
    
    // if (hints) {
    //     hints->flags = PMinSize | PMaxSize;
        
    //     hints->min_width  = SCREEN_WIDTH;
    //     hints->max_width  = SCREEN_WIDTH;
    //     hints->min_height = SCREEN_HEIGHT;
    //     hints->max_height = SCREEN_HEIGHT;
        
    //     XSetWMNormalHints(display, window, hints);
        
    //     XFree(hints);
    // }

    /* ловим события окна */
    XSelectInput(display, window, ExposureMask | KeyPressMask | KeyReleaseMask | 
        StructureNotifyMask | ButtonPressMask | ButtonReleaseMask | 
        PointerMotionMask | StructureNotifyMask);

    XMapWindow(display, window);
    XStoreName(display, window, "Core");

    XEvent event;

    XShmCompletionEvent *shm_ev = (XShmCompletionEvent *)&event;

    /* создаем кисть */
    GC gc = XCreateGC(display, window, 0, NULL);

    /* событие изменения размера окна */
    int shm_completion_event_type = XShmGetEventBase(display);

    /* создаем общий буфер */
    XShmSegmentInfo shminfo[2];
    XImage *x_image[2];
    bool is_buffer_ready[2] = {true, true};
    short back_buffer_idx = 0; 

    for (short i = 0; i < 2; i++){

        x_image[i] = XShmCreateImage(display, DefaultVisual(display, screen), DefaultDepth(display, screen), ZPixmap, NULL, &shminfo[i], SCREEN_WIDTH, SCREEN_HEIGHT);

        shminfo[i].shmid = shmget(IPC_PRIVATE, x_image[i]->bytes_per_line * x_image[i]->height, IPC_CREAT | 0777);

        shminfo[i].shmaddr = (char *)shmat(shminfo[i].shmid, 0, 0);
        x_image[i]->data = shminfo[i].shmaddr;
        shminfo[i].readOnly = False;

        XShmAttach(display, &shminfo[i]);

        /* помечаем на удаление при завершении программы */
        shmctl(shminfo[i].shmid, IPC_RMID, 0);
    }

    /* создаем ид события */
    Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    Atom wm_state = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wm_fullscreen = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);

    XEvent xev;
    xev.type = ClientMessage;
    xev.xclient.window = window;
    xev.xclient.message_type = wm_state;
    xev.xclient.format = 32;

    xev.xclient.data.l[0] = 1;
    xev.xclient.data.l[1] = wm_fullscreen;
    xev.xclient.data.l[2] = 0;
    xev.xclient.data.l[3] = 1;
    xev.xclient.data.l[4] = 0; 

    /* отключаем автоповтор */
    XAutoRepeatOff(display);

    /* создаем обьекты для графики */

    short move_x, move_y, rot, mouse_x, mouse_y;

    uint8_t *fb = (uint8_t*)aligned_alloc(32, (SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t) + 31) & ~31);
    memset(fb, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));

    Rect player_1 = {900, 150, {0, 0, 0, 0}, {0, 0, 0, 0}, 200, 200, alpha_writer(0xbb0505, 0.5f), 100.0f, {3, 10}};
    Rect player_2 = {900, 150, {400, 0, 0, 0}, {500, 0, 0, 0}, 200, 200, alpha_writer(0xbb0500, 0.5f), 100.0f, {3, 10}};
    
    // int n = 100;
    // float *arr_x = (float *)malloc((n + 1) * sizeof(float));
    // float *arr_y = (float *)malloc((n + 1) * sizeof(float));

    // arr_x[n] = 500.0f;
    // arr_y[n] = 400.0f;
    // init_poligon(arr_x, arr_y, n, arr_x[n], arr_y[n], 100);

    init_rect(&player_1);
    init_rect(&player_2);

    // Circle player_1 = {500, 200, 100, 100, 50, 0xbb0505};

    Text fps_text = {0, 2, 10, 10, L"FPS:", alpha_writer(0x2dc100, 0.5f), 8, {0, 0}};

    //Line simple_line = {10, 10, 100, 150, 500, 0xeedb04};

    /* инициализирум счетчик фпс */
    const double fps = 10000.0;

    double last_frame_time = get_time_in_seconds();
    float delta_time = 0.0f;

    float fps_timer = 0.0f;
    unsigned short fps_count = 0;
    unsigned short current_fps = 0;

    const double target_frame_time = 1.0 / fps;

    // WavHeader my_header = {0};
    // short *sound_data = load_wav("Linux/2D/музон.wav", &my_header); 

    /* создаем парлельность */
    #pragma omp parallel
    {
    /* делаем основной поток */
    #pragma omp single
    {

    /* можно в начале проиграть музон */

    // if (sound_data) {
    //     apply_volume(sound_data, my_header.subchunk2_size, 0.6f);
    //     play_wav(sound_data, &my_header, handle);
    // }

    /* загрузка картинки */
    // TGA_sprite my_sprite = {0};

    // my_sprite.x = 50;
    // my_sprite.y = 50;
    // my_sprite.speed = 400;

    // load_tga("Linux/2D/картинка.tga", &my_sprite);

    // transparent_pixels(&my_sprite, 255, 255, 255, 0);

    while (true) {

        /* счетчик фпс */
        double current_frame_time = get_time_in_seconds();
        delta_time = current_frame_time - last_frame_time;
        last_frame_time = current_frame_time;

        delta_time = delta_time > 0.1f ? 0.1f : delta_time;

        fps_timer += delta_time;
        fps_count++;
        if (fps_timer >= 1.0f) {
            current_fps = fps_count;
            swprintf(fps_text.text, 32, L"FPS:%d", current_fps);
            fps_count = 0;
            fps_timer -= 1.0f;

            // char str[10];
            // wcstombs(str, fps_text.text, sizeof(str));
            // XStoreName(display, window, str);
        }
        
        /* ловим и обрабатываем события окна*/
        while (XPending(display)) {

            XNextEvent(display, &event);

            /* события готовности кадра на отправку */
            if (event.type == shm_completion_event_type) {
                if (shm_ev->shmseg == shminfo[0].shmseg) is_buffer_ready[0] = true;
                else if (shm_ev->shmseg == shminfo[1].shmseg) is_buffer_ready[1] = true;
            }

            /* событие изменения размера экрана */
            if (event.type == ConfigureNotify){
                int new_width = event.xconfigure.width;
                int new_height = event.xconfigure.height;

                if (new_width != SCREEN_WIDTH || new_height != SCREEN_HEIGHT) {
                    if (new_width <= 0) new_width = 1;
                    if (new_height <= 0) new_height = 1;
                    SCREEN_WIDTH = new_width;
                    SCREEN_HEIGHT = new_height;

                    XSync(display, False);

                    for (short i = 0; i < 2; i++){

                        XShmDetach(display, &shminfo[i]);
                        XDestroyImage(x_image[i]);
                        shmdt(shminfo[i].shmaddr);

                        x_image[i] = XShmCreateImage(display, DefaultVisual(display, screen), DefaultDepth(display, screen), ZPixmap, NULL, &shminfo[i], SCREEN_WIDTH, SCREEN_HEIGHT);

                        shminfo[i].shmid = shmget(IPC_PRIVATE, x_image[i]->bytes_per_line * x_image[i]->height, IPC_CREAT | 0777);

                        shminfo[i].shmaddr = (char *)shmat(shminfo[i].shmid, 0, 0);
                        x_image[i]->data = shminfo[i].shmaddr;
                        shminfo[i].readOnly = False;

                        shmctl(shminfo[i].shmid, IPC_RMID, 0);

                        XShmAttach(display, &shminfo[i]);

                        is_buffer_ready[i] = true;
                    }

                    free(framebuffer_2);
                    framebuffer_2 = (uint32_t*)aligned_alloc(32, (SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint32_t) + 31) & ~31);

                    fb = (uint8_t *)realloc(fb, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));
                    memset(fb, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));

                    back_buffer_idx = 0; 
                }
            }

            /* события выхода */
            if (event.type == ClientMessage){
                if ((Atom)event.xclient.data.l[0] == wm_delete_window){
                    if (handle){
                        snd_pcm_drain(handle);
                        snd_pcm_close(handle);
                    }

                    XAutoRepeatOn(display);

                    XDestroyWindow(display, window);

                    for (short i = 0; i < 2; i++){
                        XShmDetach(display, &shminfo[i]);
                        XDestroyImage(x_image[i]);
                        shmdt(shminfo[i].shmaddr); 
                    }

                    XFreeGC(display, gc);
                    XAutoRepeatOn(display);
                    XCloseDisplay(display);

                    // free(framebuffer_2);

                    exit(0);
                }
            }

            /* клавиша нажата */
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                keys[key] = key < 65536;
            }

            /* клавиша разжата */
            if (event.type == KeyRelease) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                keys[key] = key >= 65536;
            }

            if (event.type == MotionNotify){
                mouse_x = event.xmotion.x;
                mouse_y = event.xmotion.y;
            }

            if (event.type == ButtonPress){
                buttons[event.xbutton.button] = event.xbutton.button < 8;
            }

            if (event.type == ButtonRelease){
                buttons[event.xbutton.button] = event.xbutton.button >= 8;
            }
        }

        /* сам игровой цикл */
        move_x = keys[XK_Right] - keys[XK_Left];
        move_y = keys[XK_Down] - keys[XK_Up];

        rot = (keys[XK_d] - keys[XK_a]) + (keys[XK_D] - keys[XK_A]);

        float d_x = move_x * player_1.speed * delta_time, d_y = move_y * player_1.speed * delta_time;

        if (buttons[Button1]){
            d_x += mouse_x - (player_1.x[0] + player_1.x[2]) * 0.5f;
            d_y += mouse_y - (player_1.y[0] + player_1.y[2]) * 0.5f;
            // buttons[Button1] = false;
        }

        bool a, b, c, d, a2, b2, c2, d2;

        /* обработка столкновений */

        a = is_in_box(player_2.x, player_2.y, player_1.x[0] + d_x, player_1.y[0] + d_y);
        b = is_in_box(player_2.x, player_2.y, player_1.x[1] + d_x, player_1.y[1] + d_y);
        c = is_in_box(player_2.x, player_2.y, player_1.x[2] + d_x, player_1.y[2] + d_y);
        d = is_in_box(player_2.x, player_2.y, player_1.x[3] + d_x, player_1.y[3] + d_y);

        a2 = is_in_box(player_1.x, player_1.y, player_2.x[0], player_2.y[0]);
        b2 = is_in_box(player_1.x, player_1.y, player_2.x[1], player_2.y[1]);
        c2 = is_in_box(player_1.x, player_1.y, player_2.x[2], player_2.y[2]);
        d2 = is_in_box(player_1.x, player_1.y, player_2.x[3], player_2.y[3]);

        if (!(a || b || c || d || a2 || b2 || c2 || d2)) {
            vector_add_scal(player_1.x, 4, d_x);
            vector_add_scal(player_1.y, 4, d_y);
        }

        

        if (rot) rotate_polygon(player_1.x, player_1.y, 4, (M_PI / 180.0f) * rot * player_1.deg * delta_time);

        if (keys[XK_F11]){
            xev.xclient.data.l[0] = 1;
            XSendEvent(display,  DefaultRootWindow(display), False, SubstructureNotifyMask | SubstructureRedirectMask, &xev);
            keys[XK_F11] = false;
        }

        if (keys[XK_F10]) {
            xev.xclient.data.l[0] = 0;
            XSendEvent(display,  DefaultRootWindow(display), False, SubstructureNotifyMask | SubstructureRedirectMask, &xev);
            keys[XK_F10] = false;
        }

        if (keys[XK_F12]){
            save_tga("картинка.tga", SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer);
            keys[XK_F12] = false;
        }

        // short scale_dir = keys[XK_0] - keys[XK_9];

        // if (scale_dir != 0) {
        //     float factor = (scale_dir > 0) ? (player_1.scale * delta_time) : reciprocal(player_1.scale * delta_time);
        //     player_1.width *= factor;
        //     player_1.height *= factor;
        // }

        /* отрисовка кадров */
        if (is_buffer_ready[back_buffer_idx]){
            framebuffer = (uint32_t*)x_image[back_buffer_idx]->data;

            clear_screen_avx(0x1A1A2E, SCREEN_WIDTH * SCREEN_HEIGHT, framebuffer);
            // clear_screen(0x1A1A2E, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer); // (HEX: #1A1A2E)
            // memset(framebuffer, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint32_t));

            draw_rect_glass(player_2.x[0], player_2.y[0], player_2.width, player_2.height, player_2.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer);

            draw_triangle_avx_glass(player_1.x[0], player_1.y[0], player_1.x[1], player_1.y[1], 
                player_1.x[2], player_1.y[2], player_1.color,  SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer, fb);

            draw_triangle_avx_glass(player_1.x[2], player_1.y[2], player_1.x[3], player_1.y[3],
                player_1.x[0], player_1.y[0], player_1.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer, fb);

            memset(fb, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));
            //memcpy_avx_epi32(framebuffer, framebuffer_2, SCREEN_WIDTH * SCREEN_HEIGHT);
            
            draw_string_glass(fps_text.x, fps_text.y, fps_text.text, fps_text.scale, fps_text.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer); // (HEX: #13b17c)

            is_buffer_ready[back_buffer_idx] = false;

            XShmPutImage(display, window, gc, x_image[back_buffer_idx], 0, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, true);
            XFlush(display);

            back_buffer_idx = 1 - back_buffer_idx; 
        }

        /* задержка времени для фпс */
        double frame_time = get_time_in_seconds() - current_frame_time;

        if (frame_time < target_frame_time) {   
            struct timespec ts = {0, (long)((target_frame_time - frame_time) * 1000000000.0)};
            nanosleep(&ts, NULL);
        }
    }
    }
    }
    return 0;
}
