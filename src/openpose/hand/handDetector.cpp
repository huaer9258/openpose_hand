#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/hand/handDetector.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
using namespace caffe;  // NOLINT(build/namespaces)


class Detector {
public:
    Detector(const string& model_file,
             const string& weights_file,
             const string& mean_file,
             const string& mean_value);
    
    std::vector<vector<float> > Detect(const cv::Mat& img);
    
private:
    void SetMean(const string& mean_file, const string& mean_value);
    
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    
    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
    
private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif
    
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    
    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    
    Preprocess(img, &input_channels);
    
    net_->Forward();
    
    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
        "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
        
        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";
        
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }
        
        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);
        
        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
        "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
        "Specify either 1 mean_value or as many as channels: " << num_channels_;
        
        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
    
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
/*
DEFINE_string(mean_file, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
              "If specified, can be one value or can be same as image channels"
              " - would subtract from the corresponding channel). Separated by ','."
              "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
              "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.7,
              "Only store detections with score higher than the threshold.");

// int main(int argc, char** argv) {
//   ::google::InitGoogleLogging(argv[0]);
//   // Print output to stderr (while still logging)
//   FLAGS_alsologtostderr = 1;

// #ifndef GFLAGS_GFLAGS_H_
//   namespace gflags = google;
// #endif

//   gflags::SetUsageMessage("Do detection using SSD mode.\n"
//         "Usage:\n"
//         "    ssd_detect [FLAGS] model_file weights_file list_file\n");
//   gflags::ParseCommandLineFlags(&argc, &argv, true);

//   if (argc < 4) {
//     gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
//     return 1;
//   }

const string& model_file = "";//argv[1];
const string& weights_file = "";//argv[2];
const string& mean_file ="";//FLAGS_mean_file;
const string& mean_value ="104,117,123";//FLAGS_mean_value;
const string& file_type = "image";
const string& out_file = "./score.txt";//FLAGS_out_file;
const float confidence_threshold = 0.7;//FLAGS_confidence_threshold;

// Initialize the network.
Detector detector(model_file, weights_file, mean_file, mean_value);

// Set the output mode.
std::streambuf* buf = std::cout.rdbuf();
std::ofstream outfile;
if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
        buf = outfile.rdbuf();
    }
}
std::ostream out(buf);

// // Process image one by one.
// std::ifstream infile(argv[3]);
// std::string file;
// while (infile >> file) {
if (file_type == "image") {
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    std::vector<vector<float> > detections = detector.Detect(img);
 */

    /* Print the detection results. */
    // for (int i = 0; i < detections.size(); ++i) {
    //     const vector<float>& d = detections[i];
    //     // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    //     CHECK_EQ(d.size(), 7);
    //     const float score = d[2];
    //     if (score >= confidence_threshold) {
    //         out << file << " ";
    //         out << static_cast<int>(d[1]) << " ";
    //         out << score << " ";
    //         out << static_cast<int>(d[3] * img.cols) << " ";
    //         out << static_cast<int>(d[4] * img.rows) << " ";
    //         out << static_cast<int>(d[5] * img.cols) << " ";
    //         out << static_cast<int>(d[6] * img.rows) << std::endl;
    //         }
    //         }
            //}
            //   else if (file_type == "video") {
            //     cv::VideoCapture cap(file);
            //     if (!cap.isOpened()) {
            //       LOG(FATAL) << "Failed to open video: " << file;
            //     }
            //     cv::Mat img;
            //     int frame_count = 0;
            //     while (true) {
            //       bool success = cap.read(img);
            //       if (!success) {
            //         LOG(INFO) << "Process " << frame_count << " frames from " << file;
            //         break;
            //       }
            //       CHECK(!img.empty()) << "Error when read frame";
            //       std::vector<vector<float> > detections = detector.Detect(img);
            
            //       /* Print the detection results. */
            //       for (int i = 0; i < detections.size(); ++i) {
            //         const vector<float>& d = detections[i];
            //         // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            //         CHECK_EQ(d.size(), 7);
            //         const float score = d[2];
            //         if (score >= confidence_threshold) {
            //           out << file << "_";
            //           out << std::setfill('0') << std::setw(6) << frame_count << " ";
            //           out << static_cast<int>(d[1]) << " ";
            //           out << score << " ";
            //           out << static_cast<int>(d[3] * img.cols) << " ";
            //           out << static_cast<int>(d[4] * img.rows) << " ";
            //           out << static_cast<int>(d[5] * img.cols) << " ";
            //           out << static_cast<int>(d[6] * img.rows) << std::endl;
            //         }
            //       }
            //       ++frame_count;
            //     }
            //     if (cap.isOpened()) {
            //       cap.release();
            //     }
            //   } else {
            //     LOG(FATAL) << "Unknown file_type: " << file_type;
            //   }
            // }
            //   return 0;
            // }
            // #else
            // int main(int argc, char** argv) {
            //   LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
            // }
            // #endif  // USE_OPENCV
            
            

namespace op
{
	inline Rectangle<float> getHandFromPoseIndexes(const Array<float>& poseKeypoints, const unsigned int person, const unsigned int wrist,
	                                               const unsigned int elbow, const unsigned int shoulder, const float threshold)
	{
	    try
	    {
	        Rectangle<float> handRectangle;
	        // Parameters
	        const auto* posePtr = &poseKeypoints.at(person*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
	        const auto wristScoreAbove = (posePtr[wrist*3+2] > threshold);
	        const auto elbowScoreAbove = (posePtr[elbow*3+2] > threshold);
	        const auto shoulderScoreAbove = (posePtr[shoulder*3+2] > threshold);
	        const auto ratioWristElbow = 0.33f;
	        // Hand
	        if (wristScoreAbove && elbowScoreAbove && shoulderScoreAbove)
	        {
	            // pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
	            handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
	            handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
	            const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
	            const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
	            handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
	        }
	        // height = width
	        handRectangle.height = handRectangle.width;
	        // x-y refers to the center --> offset to topLeft point
	        handRectangle.x -= handRectangle.width / 2.f;
	        handRectangle.y -= handRectangle.height / 2.f;
	        // Return result
	        return handRectangle;
	    }
	    catch (const std::exception& e)
	    {
	        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	        return Rectangle<float>{};
	    }
	}

	inline std::array<Rectangle<float>, 2> getHandFromPoseIndexes(const Array<float>& poseKeypoints, const unsigned int person,
	                                                              const unsigned int lWrist, const unsigned int lElbow, const unsigned int lShoulder,
	                                                              const unsigned int rWrist, const unsigned int rElbow, const unsigned int rShoulder,
	                                                              const float threshold)
	{
	    try
	    {
	        return {getHandFromPoseIndexes(poseKeypoints, person, lWrist, lElbow, lShoulder, threshold),
	            getHandFromPoseIndexes(poseKeypoints, person, rWrist, rElbow, rShoulder, threshold)};
	    }
	    catch (const std::exception& e)
	    {
	        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	        return std::array<Rectangle<float>, 2>(); // Parentheses instead of braces to avoid error in GCC 4.8
	    }
	}

	float getAreaRatio(const Rectangle<float>& rectangleA, const Rectangle<float>& rectangleB)
	{
	    try
	    {
	        // https://stackoverflow.com/a/22613463
	        const auto sA = rectangleA.area();
	        const auto sB = rectangleB.area();
	        const auto bottomRightA = rectangleA.bottomRight();
	        const auto bottomRightB = rectangleB.bottomRight();
	        const auto sI = fastMax(0.f, 1.f + fastMin(bottomRightA.x, bottomRightB.x) - fastMax(rectangleA.x, rectangleB.x))
	        * fastMax(0.f, 1.f + fastMin(bottomRightA.y, bottomRightB.y) - fastMax(rectangleA.y, rectangleB.y));
	        // // Option a - areaRatio = 1.f only if both Rectangle has same size and location
	        // const auto sU = sA + sB - sI;
	        // return sI / (float)sU;
	        // Option b - areaRatio = 1.f if at least one Rectangle is contained in the other
	        const auto sU = fastMin(sA, sB);
	        return fastMin(1.f, sI / (float)sU);
	    }
	    catch (const std::exception& e)
	    {
	        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	        return 0.f;
	    }
	}

	void trackHand(Rectangle<float>& currentRectangle, const std::vector<Rectangle<float>>& previousHands)
	{
	    try
	    {
	        if (currentRectangle.area() > 0 && previousHands.size() > 0)
	        {
	            // Find closest previous rectangle
	            auto maxIndex = -1;
	            auto maxValue = 0.f;
	            for (auto previous = 0 ; previous < previousHands.size() ; previous++)
	            {
	                const auto areaRatio = getAreaRatio(currentRectangle, previousHands[previous]);
	                if (maxValue < areaRatio)
	                {
	                    maxValue = areaRatio;
	                    maxIndex = previous;
	                }
	            }
	            // Update current rectangle with closest previous rectangle
	            if (maxIndex > -1)
	            {
	                const auto& prevRectangle = previousHands[maxIndex];
	                const auto ratio = 2.f;
	                const auto newWidth = fastMax((currentRectangle.width * ratio + prevRectangle.width) * 0.5f,
	                                              (currentRectangle.height * ratio + prevRectangle.height) * 0.5f);
	                currentRectangle.x = 0.5f * (currentRectangle.x + prevRectangle.x + 0.5f * (currentRectangle.width + prevRectangle.width) - newWidth);
	                currentRectangle.y = 0.5f * (currentRectangle.y + prevRectangle.y + 0.5f * (currentRectangle.height + prevRectangle.height) - newWidth);
	                currentRectangle.width = newWidth;
	                currentRectangle.height = newWidth;
	            }
	        }
	    }
	    catch (const std::exception& e)
	    {
	        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	    }
	}

	HandDetector::HandDetector(const PoseModel poseModel) :
	// Parentheses instead of braces to avoid error in GCC 4.8
	mPoseIndexes(getPoseKeypoints(poseModel, {"LWrist", "LElbow", "LShoulder", "RWrist", "RElbow", "RShoulder"})),
	mCurrentId{0}
	{
	}

	std::vector<std::array<Rectangle<float>, 2>> HandDetector::detectHands(const cv::Mat& cvInputData, const float scaleInputToOutput) const
	{
	    try
	    {
	        const string& model_file = "./models/hand/hand_deploy.prototxt";//argv[1];
	        const string& weights_file = "./models/hand/VGG_Hand_SSD_300x300_iter_25000.caffemodel";//argv[2];
	        const string& mean_file ="";//FLAGS_mean_file;
	        const string& mean_value ="104,117,123";//FLAGS_mean_value;
	        const string& file_type = "image";
	        //const string& out_file = "./score.txt";//FLAGS_out_file;
	        const float confidence_threshold = 0.7;    //FLAGS_confidence_threshold;
	        
	        // Initialize the network.
	        Detector detector(model_file, weights_file, mean_file, mean_value);
	        std::vector<std::array<Rectangle<float>, 2>> handRectangles(1);
	        if (file_type == "image") {
	            cv::Mat img = cvInputData;
	            
	            std::vector<vector<float> > detections = detector.Detect(img);
	            const auto numberhand = detections.size();
 
	            for (auto i = 0 ; i < numberhand; ++i){
	                const vector<float>& d = detections[i];
	                const float score = d[2];
					 if (score >= confidence_threshold) {
			                Rectangle<float> handRectangle;
			                handRectangle.x = (d[3] * img.cols + d[5] * img.cols)/2.f;
			                handRectangle.y = (d[4] * img.rows + d[6] * img.rows)/2.f;
			                handRectangle.width = d[5] * img.cols - d[3] * img.cols;
			                handRectangle.height = d[6] * img.rows - d[4] * img.rows;
			                handRectangle.x -= handRectangle.width / 2.f;
			                handRectangle.y -= handRectangle.height / 2.f;
			                handRectangles.at(0).at(i) = handRectangle;
			                handRectangles.at(0).at(i) /= scaleInputToOutput;
			            }
	            }
	        }
	        return handRectangles;
	    }
	    catch (const std::exception& e)
	    {
	        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	        return std::vector<std::array<Rectangle<float>, 2>>{};
	    }
	}
	    
    std::vector<std::array<Rectangle<float>, 2>> HandDetector::trackHands(const cv::Mat& cvInputData, const float scaleInputToOutput)
    {
        try
        {
            std::lock_guard<std::mutex> lock{mMutex};
            // Baseline detectHands
            auto handRectangles = detectHands(cvInputData, scaleInputToOutput);
            // If previous hands saved
            // for (auto current = 0 ; current < handRectangles.size() ; current++)
            for (auto& handRectangle : handRectangles)
            {
                // trackHand(handRectangles[current][0], mHandLeftPrevious);
                // trackHand(handRectangles[current][1], mHandRightPrevious);
                trackHand(handRectangle[0], mHandLeftPrevious);
                trackHand(handRectangle[1], mHandRightPrevious);
            }
            // Return result
            return handRectangles;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::array<Rectangle<float>, 2>>{};
        }
    }
    
    void HandDetector::updateTracker(const std::array<Array<float>, 2>& handKeypoints, const unsigned long long id)
    {
        try
        {
            std::lock_guard<std::mutex> lock{mMutex};
            if (mCurrentId < id)
            {
                mCurrentId = id;
                // Parameters
                const auto numberPeople = handKeypoints.at(0).getSize(0);
                const auto handNumberParts = handKeypoints[0].getSize(1);
                const auto thresholdRectangle = 0.25f;
                // Update pose keypoints and hand rectangles
                mPoseTrack.resize(numberPeople);
                mHandLeftPrevious.clear();
                mHandRightPrevious.clear();
                for (auto person = 0 ; person < mPoseTrack.size() ; person++)
                {
                    const auto scoreThreshold = 0.66667f;
                    // Left hand
                    if (getAverageScore(handKeypoints[0], person) > scoreThreshold)
                    {
                        const auto handLeftRectangle = getKeypointsRectangle(handKeypoints[0], person, handNumberParts, thresholdRectangle);
                        if (handLeftRectangle.area() > 0)
                            mHandLeftPrevious.emplace_back(handLeftRectangle);
                    }
                    // Right hand
                    if (getAverageScore(handKeypoints[1], person) > scoreThreshold)
                    {
                        const auto handRightRectangle = getKeypointsRectangle(handKeypoints[1], person, handNumberParts, thresholdRectangle);
                        if (handRightRectangle.area() > 0)
                            mHandRightPrevious.emplace_back(handRightRectangle);
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
    
    std::array<unsigned int, (int)HandDetector::PosePart::Size> HandDetector::getPoseKeypoints(
                                                                                               const PoseModel poseModel, const std::array<std::string, (int)HandDetector::PosePart::Size>& poseStrings
                                                                                               ) const
    {
        std::array<unsigned int, (int)PosePart::Size> poseKeypoints;
        for (auto i = 0 ; i < poseKeypoints.size() ; i++)
            poseKeypoints.at(i) = poseBodyPartMapStringToKey(poseModel, poseStrings.at(i));
        return poseKeypoints;
    }
}