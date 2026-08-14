#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb-master/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-master/stb_image_write.h"

// gcc converter.c -o tga_converter -lm -Wall -Wextra -Werror -s -Ofast -ffast-math

#pragma pack(push, 1)
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

int is_common_format(const char *path) {
    return strstr(path, ".png")  || strstr(path, ".PNG")  ||
           strstr(path, ".jpg")  || strstr(path, ".jpeg") || strstr(path, ".JPG") ||
           strstr(path, ".bmp")  || strstr(path, ".BMP");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Использование: %s <входной_файл_jpg_png_bmp> <выходной_файл.tga>\nили\n", argv[0]);
        printf("Использование: %s <входной_файл.tga> <выходной_файл.png/.jpg/.bmp>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    if (is_common_format(input_path) && (strstr(output_path, ".tga") || strstr(output_path, ".TGA"))) {
        int width, height, channels;
        uint8_t *pixels = stbi_load(input_path, &width, &height, &channels, STBI_rgb_alpha);

        if (!pixels) {
            fprintf(stderr, "Ошибка: Не удалось загрузить или декодировать файл %s\n", input_path);
            return 2;
        }

        int total_pixels = width * height;
        for (int i = 0; i < total_pixels; i++) {
            uint8_t *p = &pixels[i * 4];
            uint8_t temp = p[0];
            p[0] = p[2];
            p[2] = temp;
        }

        FILE *out_file = fopen(output_path, "wb");
        if (!out_file) {
            fprintf(stderr, "Ошибка: Не удалось создать файл %s\n", output_path);
            stbi_image_free(pixels);
            return 3;
        }

        TGAHeader header = {0};
        header.image_type = 2;
        header.width = (uint16_t)width;
        header.height = (uint16_t)height;
        header.bits_per_pixel = 32;
        header.image_descriptor = 0x20;

        fwrite(&header, sizeof(TGAHeader), 1, out_file);
        fwrite(pixels, 1, (size_t)(width * height * 4), out_file);

        fclose(out_file);
        stbi_image_free(pixels);

        printf("Успешно сконвертировано! Файл сохранен как: %s (%dx%d, 32-bit BGRA)\n", output_path, width, height);
        return 0;

    } else if ((strstr(input_path, ".tga") || strstr(input_path, ".TGA")) && is_common_format(output_path)) {
        FILE *f = fopen(input_path, "rb");
        if (!f) {
            fprintf(stderr, "Ошибка: Не удалось открыть TGA файл %s\n", input_path);
            return 1;
        }

        TGAHeader header;
        if (fread(&header, sizeof(TGAHeader), 1, f) != 1) {
            fprintf(stderr, "Ошибка чтения заголовка TGA\n");
            fclose(f);
            return 1;
        }

        if (header.bits_per_pixel != 32) {
            fprintf(stderr, "Ошибка: Этот конвертер ожидает только наш 32-битный формат TGA!\n");
            fclose(f);
            return 1;
        }

        if (header.id_length > 0) {
            fseek(f, header.id_length, SEEK_CUR);
        }

        int width = header.width;
        int height = header.height;
        int total_pixels = width * height;

        uint8_t *pixels = (uint8_t *)malloc(total_pixels * 4);
        if (!pixels) {
            fclose(f);
            return 1;
        }

        if (fread(pixels, 4, total_pixels, f) != (size_t)total_pixels) {
            fprintf(stderr, "Ошибка чтения пикселей из TGA\n");
            free(pixels);
            fclose(f);
            return 1;
        }
        fclose(f);

        for (int i = 0; i < total_pixels; i++) {
            uint8_t *p = &pixels[i * 4];
            uint8_t b = p[0];
            uint8_t r = p[2];

            p[0] = r;
            p[2] = b;

            if (p[3] == 0) {
                p[3] = 255;
            }
        }

        int success = 0;
        if (strstr(output_path, ".png") || strstr(output_path, ".PNG")) {
            success = stbi_write_png(output_path, width, height, 4, pixels, width * 4);
        }
        else if (strstr(output_path, ".jpg") || strstr(output_path, ".jpeg") || strstr(output_path, ".JPG")) {
            success = stbi_write_jpg(output_path, width, height, 4, pixels, 100);
        }
        else if (strstr(output_path, ".bmp") || strstr(output_path, ".BMP")) {
            success = stbi_write_bmp(output_path, width, height, 4, pixels);
        }

        if (success) {
            printf("Конвертация успешна! Создан файл: %s (%dx%d)\n", output_path, width, height);
        } else {
            fprintf(stderr, "Ошибка при сохранении итогового файла %s\n", output_path);
        }

        free(pixels);
        return 0;
    } else {
        printf("Неизвестные форматы файлов или неверное направление конвертации.\n");
        return 1;
    }
}
