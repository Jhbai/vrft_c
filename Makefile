CC = gcc
CFLAGS = -O3 -mavx -mavx2 -mfma -pthread -Wall
TARGET = main
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
    $(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
    rm -f $(TARGET)
