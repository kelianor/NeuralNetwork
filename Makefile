train-model:
	python3 ./src/train.py
program:
	g++ ./src/main.cpp -lSDL3 -o ./out/main
	exec ./out/main