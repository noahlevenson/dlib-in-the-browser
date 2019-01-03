// source ./emsdk_env.sh --build=Release

// emcc -msse3 -std=c++11 -O3 -I ../dlib dlib-wasm-test.cpp ../dlib/dlib/all/source.cpp -lstdc++ -lpthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4 -s TOTAL_MEMORY=1024MB -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']" -s WASM=1 -o dlib-wasm-test.js
// ^^^ this got it to compile!
// Note that I had to add #define DLIB_NO_GUI_SUPPORT to the source.cpp file
 
#include <pmmintrin.h>
#include <emscripten/emscripten.h>
#include <iostream>
#include <vector>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;
using namespace dlib;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;

EMSCRIPTEN_KEEPALIVE void init_shape_predictor(char input_buf[], uint32_t len)
{
	cout << "Deserializing model, " << len << " bytes...\n";

	std::string model(input_buf, len);
	std::istringstream model_istringstream(model);
	deserialize(sp, model_istringstream);

	cout << "deserializing done!\n";

	delete [] input_buf;
}

EMSCRIPTEN_KEEPALIVE uint16_t* detect(unsigned char input_buf[])
{
	array2d<rgb_pixel> input_image;
	input_image.set_size(480, 640);

	for (int i = 0; i < 480; i += 1) 
	{
		for (int j = 0; j < 640; j += 1) 
		{
			uint32_t offset = (i * 640 * 4) + j * 4;

			unsigned char r = input_buf[offset];
			unsigned char g = input_buf[offset + 1];
			unsigned char b = input_buf[offset + 2];

			input_image[i][j] = {r, g, b};
		}
	}

	// pyramid_up(input_image); // This upsamples the input image to better detect small faces, but it also upsamples the resolution of output data
	std::vector<rectangle> d = detector(input_image);
	// cout << "Number of faces detected: " << d.size() << endl;

	// cout << "Bounding box TL: " << d[0].tl_corner().x() << ", " << d[0].tl_corner().y() << endl;
	// cout << "Bounding box WH: " << d[0].width() << ", " << d[0].height() << endl;

	full_object_detection shape = sp(input_image, d[0]);
	// cout << "Number of landmark parts: " << shape.num_parts() << endl;

	int plen = shape.num_parts() * 2 + 1;
	uint16_t* parts = new uint16_t[plen];
	parts[0] = plen;

	for (int i = 0, j = 1; i < shape.num_parts(); i += 1, j += 2) 
	{
		parts[j] = shape.part(i).x();
		parts[j + 1] = shape.part(i).y();
	}

	return parts;
}

int main(int argc, char* argv[]) 
{
	cout << "Hello from C++ main()\n";
	return 0;
}

#ifdef __cplusplus
}
#endif