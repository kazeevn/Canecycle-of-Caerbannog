test:
	g++ -O0 -Wall -Werror --std=c++11 -lgtest_main -lgtest -lpthread src/test_loss_function.cpp -o bin/test
	bin/test
