CFLAGS = -Wall -Wextra -Werror -Ofast -s -flto

LIBS = -lX11 -lXext -lm -lasound -fopenmp -mavx2

TARGET = ./a.out

SOURCHE = Linux/main.c

all: $(TARGET)

$(TARGET):
	gcc $(SOURCHE) $(CFLAGS) $(LIBS)

clean:
	rm -rf $(TARGET)

run:
	$(TARGET)
