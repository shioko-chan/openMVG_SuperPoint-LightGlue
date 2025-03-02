// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The <cereal/archives> headers are special and must be included first.
#include <cereal/archives/json.hpp>

#include "openMVG/features/akaze/image_describer_akaze_io.hpp"

#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/loggerprogress.hpp"
#include "openMVG/system/timer.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "nonFree/sift/SIFT_describer_io.hpp"

#include <cereal/details/helpers.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class SuperPoint_Image_describer : public openMVG::features::Image_describer
{
private:
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream;

  std::vector<float> kp_h, score_h, desc_h;
  float *input = nullptr, *kp = nullptr, *score = nullptr, *desc = nullptr;
  size_t input_size, kp_size, score_size, desc_size;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_input;

  const char *input_name = "image";
  struct
  {
    const char *keypoints = "keypoints";
    const char *scores = "scores";
    const char *descriptors = "descriptors";
  } output_names;

public:
  SuperPoint_Image_describer(std::unique_ptr<nvinfer1::IExecutionContext> context) : Image_describer(), context(std::move(context))
  {
    cudaStreamCreate(&stream);
  }

  ~SuperPoint_Image_describer()
  {
    if (this->input)
      cudaFree(this->input);
    if (this->kp)
      cudaFree(this->kp);
    if (this->score)
      cudaFree(this->score);
    if (this->desc)
      cudaFree(this->desc);
    cudaStreamDestroy(stream);
  }

  bool Set_configuration_preset(openMVG::features::EDESCRIBER_PRESET preset) override
  {
    return true;
  }

  inline size_t units_size(nvinfer1::Dims &&dims)
  {
    size_t output_size = 1;
    for (int j = 0; j < dims.nbDims; ++j)
    {
      output_size *= dims.d[j];
    }
    return output_size;
  }

  void print_dims(nvinfer1::Dims &&dims, const char *name)
  {
    char *buffer = new char[256];
    std::sprintf(buffer, "%s has shape: [", name);
    for (int j = 0; j < dims.nbDims; ++j)
    {
      std::sprintf(buffer, "%s%ld, ", buffer, dims.d[j]);
    }
    std::sprintf(buffer, "%s]", buffer);
    OPENMVG_LOG_INFO << buffer;
  }

  void check_size(size_t desire_size, void **current_buffer, size_t &current_size)
  {
    if (*current_buffer == nullptr || current_size < desire_size)
    {
      if (current_buffer)
      {
        if (cudaFreeAsync(*current_buffer, this->stream) != cudaSuccess)
        {
          OPENMVG_LOG_ERROR << "Failed to free memory on device";
        }
      }
      if (cudaMallocAsync(current_buffer, desire_size, this->stream) != cudaSuccess)
      {
        OPENMVG_LOG_ERROR << "Failed to allocate memory on device";
      }
      current_size = desire_size;
    }
  }

  void check_size(size_t desire_size, std::vector<float> &current_buffer)
  {
    if (current_buffer.size() < desire_size)
    {
      current_buffer.resize(desire_size);
    }
  }

  std::unique_ptr<openMVG::features::Regions> Describe(const openMVG::image::Image<unsigned char> &image, const openMVG::image::Image<unsigned char> *mask = nullptr) override
  {
    if (img_input.rows() != image.Height() || img_input.cols() != image.Width())
    {
      OPENMVG_LOG_INFO << "!!!!!Resizing buffer to: [" << image.Height() << ", " << image.Width() << "]";
      img_input.resize(image.Height(), image.Width());
    }
    img_input = image.cast<float>();
    img_input /= 255.0f;
    const size_t input_size = img_input.size() * sizeof(float);

    this->check_size(input_size, reinterpret_cast<void **>(&this->input), this->input_size);
    cudaMemcpyAsync(this->input, img_input.data(), input_size, cudaMemcpyHostToDevice, this->stream);

    OPENMVG_LOG_INFO << "Input tensor prepared with shape: [" << img_input.rows() << ", " << img_input.cols() << "]";

    this->context->setInputShape(this->input_name, nvinfer1::Dims4(1, 1, img_input.rows(), img_input.cols()));
    this->context->setInputTensorAddress(this->input_name, reinterpret_cast<void *>(this->input));

    size_t kp_units = this->units_size(this->context->getTensorShape(this->output_names.keypoints));
    size_t score_units = this->units_size(this->context->getTensorShape(this->output_names.scores));
    size_t desc_units = this->units_size(this->context->getTensorShape(this->output_names.descriptors));

    this->print_dims(this->context->getTensorShape(this->output_names.keypoints), this->output_names.keypoints);
    this->print_dims(this->context->getTensorShape(this->output_names.scores), this->output_names.scores);
    this->print_dims(this->context->getTensorShape(this->output_names.descriptors), this->output_names.descriptors);

    this->check_size(kp_units * sizeof(float), reinterpret_cast<void **>(&this->kp), this->kp_size);
    this->check_size(score_size * sizeof(float), reinterpret_cast<void **>(&this->score), this->score_size);
    this->check_size(desc_size * sizeof(float), reinterpret_cast<void **>(&this->desc), this->desc_size);

    this->context->setOutputTensorAddress(this->output_names.keypoints, reinterpret_cast<void *>(this->kp));
    this->context->setOutputTensorAddress(this->output_names.scores, reinterpret_cast<void *>(this->score));
    this->context->setOutputTensorAddress(this->output_names.descriptors, reinterpret_cast<void *>(this->desc));

    this->context->enqueueV3(this->stream);

    this->check_size(kp_units, this->kp_h);
    this->check_size(score_units, this->score_h);
    this->check_size(desc_units, this->desc_h);

    cudaMemcpyAsync(this->kp_h.data(), this->kp, kp_units * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(this->score_h.data(), this->score, score_units * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(this->desc_h.data(), this->desc, desc_units * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);

    OPENMVG_LOG_INFO << "Inference completed successfully";

    const int num_keypoints = kp_units / 2;

    if (num_keypoints == 0)
    {
      OPENMVG_LOG_WARNING << "No keypoints detected!";
      return std::make_unique<openMVG::features::SuperPoint_Regions>();
    }

    auto regions = std::make_unique<openMVG::features::SuperPoint_Regions>();
    regions->Features().reserve(num_keypoints);
    regions->Descriptors().reserve(num_keypoints);

    OPENMVG_LOG_INFO << "Number of keypoints detected: " << num_keypoints;

    for (int i = 0; i < num_keypoints; ++i)
    {
      const float x = this->kp_h[i * 2];
      const float y = this->kp_h[i * 2 + 1];
      regions->Features().emplace_back(x, y);

      openMVG::features::SuperPoint_Regions::DescriptorT descriptor;
      const float *desc_start = this->desc_h.data() + i * 256;
      std::copy(desc_start, desc_start + 256, descriptor.data());
      regions->Descriptors().push_back(descriptor);
    }

    OPENMVG_LOG_INFO << "Feature extraction completed";

    return regions;
  }

  std::unique_ptr<openMVG::features::Regions>
  Allocate() const override
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
  Logger logger;

  std::unique_ptr<nvinfer1::ICudaEngine> engine;

public:
  NVInferEnv(const std::string &model_path = "/models/superpoint.onnx") : logger(nvinfer1::ILogger::Severity::kINFO)
  {
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

    if (
        !profile->setDimensions("image", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 64, 64)) ||
        !profile->setDimensions("image", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 456, 684)) ||
        !profile->setDimensions("image", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 3648, 5472)))
    {
      OPENMVG_LOG_ERROR << "Failed to set optimization profile dimensions";
    }

    const int32_t profile_idx = config->addOptimizationProfile(profile);
    if (profile_idx == -1)
    {
      OPENMVG_LOG_ERROR << "Failed to add optimization profile";
    }
    else
    {
      OPENMVG_LOG_INFO << "Optimization profile added with index: " << profile_idx;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4 * (static_cast<size_t>(1) << 30)); // 4GB

    this->engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  }

  std::unique_ptr<SuperPoint_Image_describer> create_describer()
  {
    return std::unique_ptr<SuperPoint_Image_describer>(new SuperPoint_Image_describer(std::unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext())));
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

/// - Compute view image description (feature & descriptor extraction)
/// - Export computed data
int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  bool bUpRight = false;
  bool bForce = false;
  std::string sFeaturePreset = "";
#ifdef OPENMVG_USE_OPENMP
  int iNumThreads = 0;
#endif

  // required
  cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
  cmd.add(make_option('o', sOutDir, "outdir"));
  // Optional
  cmd.add(make_option('u', bUpRight, "upright"));
  cmd.add(make_option('f', bForce, "force"));
  cmd.add(make_option('p', sFeaturePreset, "describerPreset"));

#ifdef OPENMVG_USE_OPENMP
  cmd.add(make_option('n', iNumThreads, "numThreads"));
#endif

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
        << "   ULTRA: !!Can take long time!!\n"
#ifdef OPENMVG_USE_OPENMP
        << "[-n|--numThreads] number of parallel computations\n"
#endif
        ;

    OPENMVG_LOG_ERROR << s;
    return EXIT_FAILURE;
  }

  OPENMVG_LOG_INFO
      << " You called : " << "\n"
      << argv[0] << "\n"
      << "--input_file " << sSfM_Data_Filename << "\n"
      << "--outdir " << sOutDir << "\n"
      << "--upright " << bUpRight << "\n"
      << "--describerPreset " << (sFeaturePreset.empty() ? "NORMAL" : sFeaturePreset) << "\n"
      << "--force " << bForce << "\n"
#ifdef OPENMVG_USE_OPENMP
      << "--numThreads " << iNumThreads << "\n"
#endif
      ;

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

    NVInferEnv env;

    const unsigned int nb_max_thread = omp_get_max_threads();

    if (iNumThreads > 0)
    {
      omp_set_num_threads(iNumThreads);
    }
    else
    {
      omp_set_num_threads(nb_max_thread);
    }

#pragma omp parallel
    {
      std::unique_ptr<Image_describer> image_describer = env.create_describer();
#pragma omp for schedule(dynamic) private(imageGray)
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
          if (!ReadImage(sView_filename.c_str(), &imageGray))
            continue;

          // Compute features and descriptors and export them to files
          auto regions = image_describer->Describe(imageGray);
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
    }
    OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();
  }
  return EXIT_SUCCESS;
}
