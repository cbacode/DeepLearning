object = Array.o ThirdArray.o read.o basic.o InputLayer.o ConvLayer.o LinearLayer.o ThirdReLULayer.o ReLULayer.o SigmoidLayer.o SpanLayer.o ThirdSpanLayer.o SoftMaxLayer.o MaxPoolingLayer.o OutputLayer.o OutputPictureLayer.o BatchLayer.o main.o
main: $(object)
	g++ $(object) -o main -O3
	
%.o : %.cpp
	g++ -c $< -O3

clean:
	rm *.o
	rm main
