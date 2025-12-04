all:
	g++ ./src/main.cpp -lSDL3 -o ./out/main
	exec ./out/main