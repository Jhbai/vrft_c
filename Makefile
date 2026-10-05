CC = gcc
CFLAGS = -O3 -mavx -mavx2 -mfma -pthread -Wall
LDFLAGS = -lcjson
TARGET = main
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
    $(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
    rm -f $(TARGET)
