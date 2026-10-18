gcc -O3 -march=native -mavx -mfma -Wall -Wextra main.c -o main -lpthread -lm
./main 0.0.0.0 9090
