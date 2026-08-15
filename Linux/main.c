#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <X11/Xutil.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <alsa/asoundlib.h>
#include "../core_2d.h"

// gcc Linux/main.c -Wall -Wextra -Werror -Ofast -s -flto -lX11 -lXext -lm -lasound -fopenmp -mavx2

#define SAMPLE_RATE 44100
#define LATENT 50000

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

    short rot;

    uint8_t *fb = calloc(sizeof(uint8_t), SCREEN_WIDTH * SCREEN_HEIGHT);

    // Rect player_1 = {900, 150, {0, 0, 0, 0}, {0, 0, 0, 0}, 200, 200, alpha_writer(0xbb0505, 0.5f), 100.0f, {3, 10}};
    // Rect player_2 = {900, 150, {400, 0, 0, 0}, {500, 0, 0, 0}, 200, 200, alpha_writer(0xbb0500, 0.5f), 100.0f, {3, 10}};

    Circle player_1 = {{0, 0}, 1, {100, 100}, 100, alpha_writer(0xbb0505, 0.5f), {0, 0}, 10.2f, {900, 900}, {0, 0}, 1};
    // Circle player_2 = {{0, 0}, 1, {500, 5400}, 5000, alpha_writer(0xbb0505, 0.5f), {0, 0}, 1e+21f, {900, 900}, {0, 0}};
    Circle player_2 = {{0, 0}, 1, {500, 400}, 100, alpha_writer(0xbb0505, 0.5f), {0, 0}, 9e+17f, {900, 900}, {0, 0}, -1};

    // init_rect(&player_1);
    // init_rect(&player_2);

    // int a = 100;
    // int b = 70;
    // Vector2f *elips = (Vector2f *)malloc(sizeof(Vector2f) * ((a + b) << 2));
    // int n_elips = init_elips(a, b, 500, 500, elips);

    Text fps_text = {{0, 0}, 2, {10, 10}, L"FPS:", alpha_writer(0x2dc100, 0.5f), 8, {0, 0}};

    //Line simple_line = {10, 10, 100, 150, 500, 0xeedb04};

    /* инициализирум счетчик фпс */
    const double fps = 180.0;
    const double physics_fps = 120.0;

    double last_frame_time = get_time_in_seconds();
    double delta_time = 0.0f;
    double physics_delta_time = 0.0;

    double fps_timer = 0.0;
    unsigned short fps_count = 0;
    unsigned short current_fps = 0;

    const double target_frame_time = 1.0 / fps;
    const double physics_target_frame_time = 1.0 / physics_fps;

    Vector2f f_keys = {18000.5f, 18000.5f};
    Vector2s mouse_loc;

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

        delta_time = fmin(0.1, delta_time);

        fps_timer += delta_time;
        fps_count++;

        physics_delta_time += delta_time;

        if (fps_timer >= 1.0) {
            current_fps = fps_count;
            swprintf(fps_text.text, 32, L"FPS:%d", current_fps);
            fps_count = 0;
            fps_timer -= 1.0;

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

                    free(framebuffer_2);
                    free(fb);

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
                mouse_loc = (Vector2s){event.xmotion.x, event.xmotion.y};
            }

            if (event.type == ButtonPress){
                buttons[event.xbutton.button] = event.xbutton.button < 8;
            }

            if (event.type == ButtonRelease){
                buttons[event.xbutton.button] = event.xbutton.button >= 8;
            }
        }

        player_1.move = (Vector2s){keys[XK_Right] - keys[XK_Left], keys[XK_Down] - keys[XK_Up]};

        rot = (keys[XK_d] - keys[XK_a]) + (keys[XK_D] - keys[XK_A]);

        // player_2.move = (Vector2s){(keys[XK_D] || keys[XK_d]) - (keys[XK_A] || keys[XK_a]), (keys[XK_S] || keys[XK_s]) - (keys[XK_W] || keys[XK_w])};

        /* сам игровой цикл + расчет физики */
        while (physics_delta_time >= physics_target_frame_time) {
            // Vector2f f_v = force_graviti(&player_1.loc, &player_2.loc, player_1.mass, player_2.mass);
            Vector2f f_v = force_kulon(&player_1.loc, &player_2.loc, player_1.q, player_2.q, 5.0f);

            acceleration(&player_1.speed, player_1.mass, &player_1.max_speed, &f_keys, &player_1.move, true, physics_target_frame_time);
            acceleration(&player_1.speed, player_1.mass, &player_1.max_speed, &f_v, &(Vector2s){1, 1}, false, physics_target_frame_time);

            Vector2f d = {player_1.speed.x * physics_target_frame_time, player_1.speed.y * physics_target_frame_time};

            acceleration(&player_2.speed, player_2.mass, &player_2.max_speed, &(Vector2f){9e+20f, 9e+20f}, &player_2.move, true, physics_target_frame_time);
            acceleration(&player_2.speed, player_2.mass, &player_2.max_speed, &f_v, &(Vector2s){-1, -1}, false, physics_target_frame_time);

            Vector2f_add_(&player_2.loc, &(Vector2f){player_2.speed.x * physics_target_frame_time, player_2.speed.y * physics_target_frame_time});

            if (buttons[Button1]){
                Vector2f_add_(&d, &(Vector2f){mouse_loc.x - player_1.loc.x, mouse_loc.y - player_1.loc.y});
                // buttons[Button1] = false;
            }

            // vector_add_scal(player_1.x, 4, d_x);
            // vector_add_scal(player_1.y, 4, d_y);

            Vector2f_add_(&player_1.loc, &d);

            rot += 0;
            // if (rot) rotate_polygon(elips, n_elips, (M_PI * reciprocal(180.0f)) * rot * 100 * physics_delta_time);

            /* обработка столкновений */
            if (is_circle_colission(&player_1.loc.x, &player_1.loc.y, &player_2.loc.x, &player_2.loc.y, player_1.r, player_2.r)){
                circle_colission(&player_1.loc.x, &player_1.loc.y, &player_2.loc.x, &player_2.loc.y, player_1.r, player_2.r, player_1.mass, player_2.mass);
                float q_general = (player_1.q + player_2.q) * 0.5f;
                player_1.q = q_general;
                player_2.q = q_general;
            }

            physics_delta_time -= physics_target_frame_time;
        }

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

        /* отрисовка кадров */
        if (is_buffer_ready[back_buffer_idx]){
            framebuffer = (uint32_t*)x_image[back_buffer_idx]->data;

            clear_screen_avx(0x1A1A2E, SCREEN_WIDTH * SCREEN_HEIGHT, framebuffer);
            // clear_screen(0x1A1A2E, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer); // (HEX: #1A1A2E)
            // memset(framebuffer, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint32_t));

            draw_filled_circle_glass(player_2.loc.x, player_2.loc.y, player_2.r, player_2.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer, fb);

            // memset(fb, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));

            draw_filled_circle_glass(player_1.loc.x, player_1.loc.y, player_1.r, player_1.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer, fb);

            memset(fb, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(uint8_t));
            //memcpy_avx_epi32(framebuffer, framebuffer_2, SCREEN_WIDTH * SCREEN_HEIGHT);

            // render_mandelbrot(SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer);

            draw_string_glass(fps_text.loc.x, fps_text.loc.y, fps_text.text, fps_text.scale, fps_text.color, SCREEN_WIDTH, SCREEN_HEIGHT, framebuffer); // (HEX: #13b17c)

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
