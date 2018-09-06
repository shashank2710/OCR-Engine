/*
 * main.cpp
 *
 *  Created on: Aug 21, 2018
 *      Author: ssatyanarayana
 */


#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;


const int64 tf_height = 128;
const int64 tf_width = 128;

int main()
{
	string folderpath = "/home/ssatyanarayana/OCR-Engine-Local/data/alphabets/myTest/*.jpg";
	std::vector<String> filenames;
	cv::glob(folderpath, filenames);

	// Set input & output nodes names
	std::string inputLayer = "conv2d_1_input";
	std::string outputLayer = "k2tfout_0";

	//Initializing the Graph
	tensorflow::GraphDef graph_def;

	// Specify file location of Output Graph
	std::string graphFile = "/home/ssatyanarayana/OCR-Engine-Local/output_graph.pb";

	// Loading the graph to the given variable
	tensorflow::Status graphLoadedStatus = ReadBinaryProto(tensorflow::Env::Default(),graphFile,&graph_def);
	if (!graphLoadedStatus.ok()){
		std::cout << graphLoadedStatus.ToString()<<std::endl;
		return 1;
	}

	std::vector<Tensor> outputs;

	for (size_t i=0; i<filenames.size(); i++)
	{
		cv::Mat image = cv::imread(filenames[i]);

		Tensor image_tensor (tensorflow::DT_FLOAT, tensorflow::TensorShape{1,tf_height,tf_width,3});
		image.convertTo(image, CV_32FC3);
		tensorflow::StringPiece tmp_data = image_tensor.tensor_data();
		memcpy(const_cast<char*>(tmp_data.data()), (image.data), tf_height * tf_width * sizeof(float));

		// Creating a Session with the Graph
		std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
		//session->tensorflow::reset(tensorflow::NewSession(tensorflow::SessionOptions()));
		tensorflow::Status session_create_status = session->Create(graph_def);

		std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{inputLayer, image_tensor}};
		outputs.clear();

		Status runStatus = session->Run(inputs, {outputLayer}, {}, &outputs);
		if (!runStatus.ok()) {
			LOG(ERROR) << "Running model failed: " << runStatus;
			return -1;
		}

		for (auto &t : outputs)
		{
			std::cout << t.DebugString()<<std::endl;
			tensorflow::TTypes<float, 2>::Tensor scores = t.flat_inner_dims<float>();
			auto dims = scores.dimensions();
			int imgCount = dims[0];
			int classesCount = dims[1];
			for(int i = 0; i<imgCount; i++)
			{
				float maxVal = scores(i,0);
				int maxIndex = 0;
				for(int j = 1; j<classesCount; j++) {
					float val = scores(i,j);
					if(val > maxVal) {
						maxVal = val;
						maxIndex = j;
					}
				}
				std::cout << "Img" << to_string(i) << " prediction: " << to_string(maxIndex) << ", score: " << to_string(maxVal)<<std::endl;
			}
		}

		cv::imshow("Original Image",image);
		cv::waitKey();
		session->Close();
		image.release();
	}
    return 0;
}

