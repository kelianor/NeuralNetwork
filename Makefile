train-model:
	python3 ./src/train.py
program-linux:
	mkdir ./out
	g++ ./src/main.cpp -lSDL3 -o ./out/main
	exec ./out/main
program-macos:
	mkdir ./out
	clang++ -o ./out/main  ./src/main.cpp -F/Library/Frameworks -framework SDL3 -Wno-narrowing -rpath /Library/Frameworks
	exec ./out/main
