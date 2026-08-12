CC = gcc
CFLAGS = -Wall -Wextra -Werror -Ofast -ffast-math -mavx2

LIBS = -lX11 -lXext -lm -lasound -fopenmp

TARGET = ./main

SOURCHE = Linux/2D/main.c

all: $(TARGET)

$(TARGET):
	$(CC) $(CFLAGS) $(SOURCHE) $(LIBS) -s -o $(TARGET)

clean:
	rm -rf $(TARGET)

run:
	$(TARGET)
