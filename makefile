object = Array.o read.o basic.o InputLayer.o ConvLayer.o LinearLayer.o ReLULayer.o SigmoidLayer.o SpanLayer.o SoftMaxLayer.o OutputLayer.o main.o
main: $(object)
	g++ $(object) -o main 
	
%.o : %.cpp
	g++ -c $< 

clean:
	rm *.o
	rm main
