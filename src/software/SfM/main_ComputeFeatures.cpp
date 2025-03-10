// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
// This file is modified by hrliu to integrate SuperPoint with OpenMVG.
// Copyright (c) 2012, 2013 Pierre MOULON.
// Copyright (c) 2025 hrliu.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The <cereal/archives> headers are special and must be included first.
#include <cereal/archives/json.hpp>

#include "openMVG/features/akaze/image_describer_akaze_io.hpp"
#include "openMVG/exif/exif_IO_EasyExif.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/loggerprogress.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/tensorrt.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "nonFree/sift/SIFT_describer_io.hpp"

#include <cereal/details/helpers.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

const size_t IMAGE_HEIGHT_LIM = 768, IMAGE_WIDTH_LIM = 960;

using namespace openMVG::TensorRT;

class SuperPoint_Image_describer : public openMVG::features::Image_describer
{
private:
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  HostAllocator input_h, kp_h, score_h, desc_h;
  GPUAllocator input;
  OutputAllocator kp, score, desc;

  cudaStream_t stream;
  const char *input_name = "image";
  struct
  {
    const char *kp = "keypoints";
    const char *score = "scores";
    const char *desc = "descriptors";
  } output_names;

public:
  SuperPoint_Image_describer(std::unique_ptr<nvinfer1::IExecutionContext> context) : Image_describer(), context(std::move(context))
  {
    cudaStreamCreate(&stream);

    this->input = GPUAllocator(stream);
    this->kp = OutputAllocator(stream);
    this->score = OutputAllocator(stream);
    this->desc = OutputAllocator(stream);

    this->context->setOptimizationProfileAsync(0, this->stream);

    this->context->setOutputAllocator(this->output_names.kp, &this->kp);
    this->context->setOutputAllocator(this->output_names.score, &this->score);
    this->context->setOutputAllocator(this->output_names.desc, &this->desc);

    this->context->setOutputTensorAddress(this->output_names.kp, nullptr);
    this->context->setOutputTensorAddress(this->output_names.score, nullptr);
    this->context->setOutputTensorAddress(this->output_names.desc, nullptr);
  }

  ~SuperPoint_Image_describer()
  {
    cudaStreamDestroy(stream);
  }

  bool Set_configuration_preset(openMVG::features::EDESCRIBER_PRESET preset) override
  {
    return true;
  }

  inline size_t units_size(nvinfer1::Dims &dims)
  {
    size_t output_size = 1;
    for (int j = 0; j < dims.nbDims; ++j)
    {
      output_size *= dims.d[j];
    }
    return output_size;
  }

  const char *print_dims(nvinfer1::Dims &&dims, const char *name)
  {
    char *buffer = new char[256];
    std::sprintf(buffer, "%s has shape: [", name);
    for (int j = 0; j < dims.nbDims; ++j)
    {
      std::sprintf(buffer, "%s%ld, ", buffer, dims.d[j]);
    }
    std::sprintf(buffer, "%s]", buffer);
    return buffer;
  }

  std::unique_ptr<openMVG::features::Regions> Describe(const openMVG::image::Image<unsigned char> &img_input, const openMVG::image::Image<unsigned char> *mask = nullptr) override
  {
    OPENMVG_LOG_ERROR << "Not implemented";
    return std::make_unique<openMVG::features::SuperPoint_Regions>();
  }

  std::unique_ptr<openMVG::features::Regions> Describe(const cv::Mat &img_input, const int64_t factor)
  {
    const int image_size = img_input.cols * img_input.rows * sizeof(float);

    std::memcpy(this->input_h.reallocate(image_size), img_input.data, image_size);
    cudaMemcpyAsync(this->input.reallocate(image_size), this->input_h.ptr, image_size, cudaMemcpyHostToDevice, this->stream);

    this->context->setInputShape(this->input_name, nvinfer1::Dims4(1, 1, img_input.rows, img_input.cols));
    this->context->setInputTensorAddress(this->input_name, this->input.ptr);

    if (!this->context->enqueueV3(this->stream))
    {
      OPENMVG_LOG_ERROR << "Failed to enqueue inference";
    }

    cudaStreamSynchronize(this->stream);

    const size_t kp_bytes = this->units_size(this->kp.output_dims) * sizeof(int64_t);
    const size_t score_bytes = this->units_size(this->score.output_dims) * sizeof(float);
    const size_t desc_bytes = this->units_size(this->desc.output_dims) * sizeof(float);

    this->kp_h.reallocate(kp_bytes);
    this->score_h.reallocate(score_bytes);
    this->desc_h.reallocate(desc_bytes);

    cudaMemcpyAsync(this->kp_h.ptr, this->kp.allocator.ptr, kp_bytes, cudaMemcpyDeviceToHost, this->stream);
    cudaMemcpyAsync(this->score_h.ptr, this->score.allocator.ptr, score_bytes, cudaMemcpyDeviceToHost, this->stream);
    cudaMemcpyAsync(this->desc_h.ptr, this->desc.allocator.ptr, desc_bytes, cudaMemcpyDeviceToHost, this->stream);

    cudaStreamSynchronize(this->stream);

    const int num_keypoints = this->units_size(this->kp.output_dims) / 2;

    if (num_keypoints == 0)
    {
      OPENMVG_LOG_WARNING << "No keypoints detected!";
      return std::make_unique<openMVG::features::SuperPoint_Regions>();
    }

    auto regions = std::make_unique<openMVG::features::SuperPoint_Regions>();
    regions->Features().reserve(num_keypoints);
    regions->Descriptors().reserve(num_keypoints);

    const float width = 1.0f * factor * img_input.cols;
    const float height = 1.0f * factor * img_input.rows;

    for (int i = 0; i < num_keypoints; ++i)
    {
      const float x = static_cast<float>(static_cast<int64_t *>(this->kp_h.ptr)[i * 2] * factor);
      const float y = static_cast<float>(static_cast<int64_t *>(this->kp_h.ptr)[i * 2 + 1] * factor);
      regions->Features().emplace_back(x, y);

      const float *desc_start = static_cast<float *>(this->desc_h.ptr) + i * 256;

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

class NVInferEnv
{
private:
  class Logger : public nvinfer1::ILogger
  {
  private:
    nvinfer1::ILogger::Severity reportableSeverity;

  public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
      if (severity > reportableSeverity)
      {
        return;
      }
      switch (severity)
      {
      case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        OPENMVG_LOG_ERROR << "[TensorRT] INTERNAL_ERROR: " << msg;
        break;
      case nvinfer1::ILogger::Severity::kERROR:
        OPENMVG_LOG_ERROR << "[TensorRT] ERROR: " << msg;
        break;
      case nvinfer1::ILogger::Severity::kWARNING:
        OPENMVG_LOG_WARNING << "[TensorRT] WARNING: " << msg;
        break;
      case nvinfer1::ILogger::Severity::kINFO:
        OPENMVG_LOG_INFO << "[TensorRT] INFO: " << msg;
        break;
      case nvinfer1::ILogger::Severity::kVERBOSE:
        OPENMVG_LOG_INFO << "[TensorRT] VERBOSE: " << msg;
        break;
      }
    }
  };
  Logger logger{nvinfer1::ILogger::Severity::kWARNING};

  std::unique_ptr<nvinfer1::ICudaEngine> engine;

public:
  using ImageSize = std::pair<size_t, size_t>; // width, height

  NVInferEnv(ImageSize max_size, ImageSize min_size, ImageSize average_size, const std::string &model_path = "/models/superpoint.onnx") : logger(nvinfer1::ILogger::Severity::kINFO)
  {
    OPENMVG_LOG_INFO << "Creating TensorRT environment";
    OPENMVG_LOG_INFO << "MAX Size: " << max_size.first << "x" << max_size.second;
    OPENMVG_LOG_INFO << "MIN Size: " << min_size.first << "x" << min_size.second;
    OPENMVG_LOG_INFO << "AVG Size: " << average_size.first << "x" << average_size.second;

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<unsigned int>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));

    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
      OPENMVG_LOG_ERROR << "Failed to parse ONNX model: " << model_path;
      return;
    }

    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();

    auto &[avg_width, avg_height] = average_size;
    auto &[min_width, min_height] = min_size;
    auto &[max_width, max_height] = max_size;

    if (!profile->setDimensions("image", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, min_height, min_width)) ||
        !profile->setDimensions("image", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, avg_height, avg_width)) ||
        !profile->setDimensions("image", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, max_height, max_width)))
    {
      OPENMVG_LOG_ERROR << "Failed to set optimization profile dimensions";
      return;
    }

    const int32_t profile_idx = config->addOptimizationProfile(profile);
    if (profile_idx == -1)
    {
      OPENMVG_LOG_ERROR << "Failed to add optimization profile";
      return;
    }
    else
    {
      OPENMVG_LOG_INFO << "Optimization profile added with index: " << profile_idx;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 * (static_cast<size_t>(1) << 30)); // 1GB

    this->engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  }

  std::unique_ptr<SuperPoint_Image_describer> create_describer()
  {
    return std::unique_ptr<SuperPoint_Image_describer>(
        new SuperPoint_Image_describer(
            std::unique_ptr<nvinfer1::IExecutionContext>(
                this->engine->createExecutionContext())));
  }
};

openMVG::features::EDESCRIBER_PRESET stringToEnum(const std::string &sPreset)
{
  openMVG::features::EDESCRIBER_PRESET preset;
  if (sPreset == "NORMAL")
    preset = openMVG::features::NORMAL_PRESET;
  else if (sPreset == "HIGH")
    preset = openMVG::features::HIGH_PRESET;
  else if (sPreset == "ULTRA")
    preset = openMVG::features::ULTRA_PRESET;
  else
    preset = openMVG::features::EDESCRIBER_PRESET(-1);
  return preset;
}

bool getImageSize(const std::string &filename, size_t &width, size_t &height)
{
  openMVG::exif::Exif_IO_EasyExif exifReader;
  if (exifReader.open(filename) && exifReader.doesHaveExifInfo())
  {
    height = exifReader.getHeight();
    width = exifReader.getWidth();
    return true;
  }
  return false;
}

/// - Compute view image description (feature & descriptor extraction)
/// - Export computed data
int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  bool bUpRight = false;
  bool bForce = false;

  // required
  cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
  cmd.add(make_option('o', sOutDir, "outdir"));
  // Optional
  cmd.add(make_option('u', bUpRight, "upright"));
  cmd.add(make_option('f', bForce, "force"));

  try
  {
    if (argc == 1)
      throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  }
  catch (const std::string &s)
  {
    OPENMVG_LOG_INFO
        << "Usage: " << argv[0] << '\n'
        << "[-i|--input_file] a SfM_Data file \n"
        << "[-o|--outdir path] \n"
        << "\n[Optional]\n"
        << "[-f|--force] Force to recompute data\n"
        << "[-u|--upright] Use Upright feature 0 or 1\n"
        << "[-p|--describerPreset]\n"
        << "  (used to control the Image_describer configuration):\n"
        << "   NORMAL (default),\n"
        << "   HIGH,\n"
        << "   ULTRA: !!Can take long time!!\n";

    OPENMVG_LOG_ERROR << s;
    return EXIT_FAILURE;
  }

  OPENMVG_LOG_INFO
      << " You called : " << "\n"
      << argv[0] << "\n"
      << "--input_file " << sSfM_Data_Filename << "\n"
      << "--outdir " << sOutDir << "\n"
      << "--upright " << bUpRight << "\n"
      << "--force " << bForce << "\n";

  if (sOutDir.empty())
  {
    OPENMVG_LOG_ERROR << "\nIt is an invalid output directory";
    return EXIT_FAILURE;
  }

  // Create output dir
  if (!stlplus::folder_exists(sOutDir))
  {
    if (!stlplus::folder_create(sOutDir))
    {
      OPENMVG_LOG_ERROR << "Cannot create output directory";
      return EXIT_FAILURE;
    }
  }

  //---------------------------------------
  // a. Load input scene
  //---------------------------------------
  openMVG::sfm::SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, openMVG::sfm::ESfM_Data(openMVG::sfm::VIEWS | openMVG::sfm::INTRINSICS)))
  {
    OPENMVG_LOG_ERROR
        << "The input file \"" << sSfM_Data_Filename << "\" cannot be read";
    return EXIT_FAILURE;
  }

  // b. Init the image_describer
  // - retrieve the used one in case of pre-computed features
  // - else create the desired one

  using namespace openMVG::features;

  // Feature extraction routines
  // For each View of the SfM_Data container:
  // - if regions file exists continue,
  // - if no file, compute features
  {
    openMVG::system::Timer timer;
    openMVG::image::Image<unsigned char> imageGray;

    openMVG::system::LoggerProgress my_progress_bar(sfm_data.GetViews().size(), "- EXTRACT FEATURES -");

    // Use a boolean to track if we must stop feature extraction
    std::atomic<bool> preemptive_exit(false);

    std::vector<NVInferEnv::ImageSize> image_sizes;
    std::vector<int64_t> factors;
    std::for_each(sfm_data.views.begin(), sfm_data.views.end(),
                  [&image_sizes, &factors, &sfm_data](const openMVG::sfm::Views::value_type &kv)
                  {
                    const std::string filename = stlplus::create_filespec(sfm_data.s_root_path, kv.second.get()->s_Img_path);
                    size_t width, height;
                    if (!getImageSize(filename, width, height))
                    {
                      OPENMVG_LOG_WARNING << "Failed to read image size from: " << filename;
                      image_sizes.emplace_back(-1, -1);
                      factors.push_back(0);
                      return;
                    }
                    for (int64_t i = 1; i <= width; ++i)
                    {
                      if (width / i <= IMAGE_WIDTH_LIM && height / i <= IMAGE_HEIGHT_LIM)
                      {
                        image_sizes.emplace_back(width / i, height / i);
                        factors.push_back(i);
                        return;
                      }
                    }
                    OPENMVG_LOG_WARNING << "Failed to find a suitable image size for: " << filename;
                    image_sizes.emplace_back(-1, -1);
                    factors.push_back(0);
                  });

    std::unordered_map<NVInferEnv::ImageSize, size_t> image_size_count;
    NVInferEnv::ImageSize min_size{IMAGE_WIDTH_LIM, IMAGE_HEIGHT_LIM}, max_size{0, 0};

    std::for_each(image_sizes.begin(), image_sizes.end(), [&min_size, &max_size, &image_size_count](const auto &image_size)
                  {
      auto &[w, h] = image_size;
      auto &[min_w, min_h] = min_size;
      auto &[max_w, max_h] = max_size;

      min_w = std::min(min_w, w);
      min_h = std::min(min_h, h);
      max_w = std::max(max_w, w);
      max_h = std::max(max_h, h);
      image_size_count[image_size]++; });

    auto [avg_size, _] = *std::max_element(image_size_count.begin(), image_size_count.end(), [](const auto &lhs, const auto &rhs)
                                           { return lhs.second < rhs.second; });

    NVInferEnv env(max_size, min_size, avg_size);

#ifdef OPENMVG_USE_OPENMP
    int thread_count = std::min(omp_get_max_threads(), 5);
    omp_set_num_threads(thread_count);
    OPENMVG_LOG_INFO << "Using " << thread_count << " threads";
#pragma omp parallel
    {
#endif
      std::unique_ptr<SuperPoint_Image_describer> image_describer = env.create_describer();
#ifdef OPENMVG_USE_OPENMP
#pragma omp for schedule(dynamic) private(imageGray)
#endif
      for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
      {
        openMVG::sfm::Views::const_iterator iterViews = sfm_data.views.begin();
        std::advance(iterViews, i);
        const openMVG::sfm::View *view = iterViews->second.get();
        const std::string
            sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path),
            sFeat = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "feat"),
            sDesc = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "desc");

        // If features or descriptors file are missing, compute them
        if (!preemptive_exit && (bForce || !stlplus::file_exists(sFeat) || !stlplus::file_exists(sDesc)))
        {
          std::vector<NVInferEnv::ImageSize>::const_iterator image_size = image_sizes.begin();
          std::vector<int64_t>::const_iterator factor = factors.begin();
          std::advance(image_size, i);
          std::advance(factor, i);

          if (*factor == 0)
            continue;

          if (!ReadImage(sView_filename.c_str(), &imageGray))
            continue;

          cv::Mat cv_image, cv_image_resized;
          cv::eigen2cv(imageGray.GetMat(), cv_image);
          cv::resize(cv_image, cv_image_resized, cv::Size(image_size->first, image_size->second), 0, 0, cv::INTER_AREA);
          cv_image_resized.convertTo(cv_image_resized, CV_32FC1, 1.0 / 255.0);

          // Compute features and descriptors and export them to files
          auto regions = image_describer->Describe(cv_image_resized, *factor);
          if (regions && !image_describer->Save(regions.get(), sFeat, sDesc))
          {
            OPENMVG_LOG_ERROR
                << "Cannot save regions for image: " << sView_filename << ';'
                << "Stopping feature extraction.";
            preemptive_exit = true;
            continue;
          }
        }
        ++my_progress_bar;
      }
#ifdef OPENMVG_USE_OPENMP
    }
#endif
    OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();
  }
  return EXIT_SUCCESS;
}
