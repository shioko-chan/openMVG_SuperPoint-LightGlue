#ifndef OPENMVG_FEATURES_SUPERPOINT_IMAGE_DESCRIBER_HPP
#define OPENMVG_FEATURES_SUPERPOINT_IMAGE_DESCRIBER_HPP

#include "openMVG/features/akaze/AKAZE.hpp"
#include "openMVG/features/image_describer.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/system/logger.hpp"

#include "openMVG/system/onnxruntime.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"

#include <unordered_map>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace openMVG
{
  using namespace openMVG::features;
  using namespace openMVG::image;

  namespace features
  {

    class SuperPoint_Image_describer : public Image_describer
    {
    private:
      std::unique_ptr<ONNXRuntime::InferEnv> infer_env;
      size_t max_h = 768, max_w = 960;

    public:
      template <class Archive>
      void serialize(Archive &ar);

      SuperPoint_Image_describer() = default;
      SuperPoint_Image_describer(size_t max_h, size_t max_w) : Image_describer(), max_h(max_h), max_w(max_w) {}

      bool Set_configuration_preset(EDESCRIBER_PRESET preset) override
      {
        OPENMVG_LOG_ERROR << "SuperPoint does not support configuration presets!";
        return false;
      }

      std::unique_ptr<Regions> Describe(const Image<unsigned char> &img_input, const Image<unsigned char> *mask = nullptr) override
      {
        if (!infer_env)
        {
          infer_env = std::make_unique<ONNXRuntime::InferEnv>("ONNX SuperPoint", "/models/superpoint.onnx");
        }

        ONNXRuntime::InferEnv &env = *infer_env;

        cv::Mat cv_image, cv_image_resized, cv_image_float;
        cv::eigen2cv(img_input.GetMat(), cv_image);

        int factor = 1;
        for (; factor < 1024; ++factor)
        {
          if (cv_image.cols / factor <= max_w && cv_image.rows / factor <= max_h)
          {
            break;
          }
        }

        cv::resize(cv_image, cv_image_resized, cv::Size(cv_image.cols / factor, cv_image.rows / factor), 0, 0, cv::INTER_AREA);

        cv_image_resized.convertTo(cv_image_float, CV_32FC1, 1.0 / 255.0);

        std::vector<float> input_data;
        if (cv_image_float.isContinuous())
        {
          input_data.assign(cv_image_float.ptr<float>(), cv_image_float.ptr<float>() + cv_image_float.total() * cv_image_float.channels());
        }
        else
        {
          input_data.reserve(cv_image_float.total() * cv_image_float.channels());
          for (int i = 0; i < cv_image_float.rows; ++i)
          {
            const float *row_ptr = cv_image_float.ptr<float>(i);
            input_data.insert(input_data.end(), row_ptr, row_ptr + cv_image_float.cols * cv_image_float.channels());
          }
        }

        env.set_input("image", input_data, {1, 1, cv_image_float.rows, cv_image_float.cols});

        std::vector<Ort::Value> res = env.infer();

        const Ort::Value &kp = res[env.get_output_index("keypoints")],
                         &score = res[env.get_output_index("scores")],
                         &desc = res[env.get_output_index("descriptors")];

        const int num_keypoints = kp.GetTensorTypeAndShapeInfo().GetShape()[1];

        if (num_keypoints == 0)
        {
          OPENMVG_LOG_WARNING << "No keypoints detected!";
          return std::make_unique<SuperPoint_Regions>();
        }

        auto regions = std::make_unique<SuperPoint_Regions>();
        regions->Features().reserve(num_keypoints);
        regions->Descriptors().reserve(num_keypoints);

        const float width = 1.0f * factor * cv_image_float.cols, height = 1.0f * factor * cv_image_float.rows;

        for (int i = 0; i < num_keypoints; ++i)
        {
          const int64_t *kp_data = kp.GetTensorData<int64_t>();
          const float x = static_cast<float>(kp_data[i * 2] * factor);
          const float y = static_cast<float>(kp_data[i * 2 + 1] * factor);
          regions->Features().emplace_back(x, y);

          const float *desc_start = desc.GetTensorData<float>() + i * 256;

          openMVG::features::SuperPoint_Regions::DescriptorT descriptor;
          descriptor.data()[0] = (x - width / 2) / (width / 2);
          descriptor.data()[1] = (y - height / 2) / (height / 2);
          std::copy(desc_start, desc_start + 256, descriptor.data() + 2);
          regions->Descriptors().push_back(descriptor);
        }

        return regions;
      }

      std::unique_ptr<openMVG::features::Regions> Allocate() const override
      {
        return std::unique_ptr<openMVG::features::SuperPoint_Regions>(new openMVG::features::SuperPoint_Regions);
      }
    };
  }

}

#endif