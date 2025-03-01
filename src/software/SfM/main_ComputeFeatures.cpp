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
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class SuperPoint_Image_describer : public openMVG::features::Image_describer
{
private:
  std::unique_ptr<Ort::Session> session_;

public:
  SuperPoint_Image_describer(const std::string &model_path)
  {
    std::cout << "SuperPoint_Image_describer" << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    this->session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  }

  bool Set_configuration_preset(openMVG::features::EDESCRIBER_PRESET preset) override
  {
    return true;
  }

  std::unique_ptr<openMVG::features::Regions> Describe(
      const openMVG::image::Image<unsigned char> &image,
      const openMVG::image::Image<unsigned char> *mask = nullptr) override
  {
    try
    {
      OPENMVG_LOG_INFO << "Starting Describe function";

      cv::Mat img;
      cv::eigen2cv(image.GetMat(), img);

      cv::Mat floatImage;
      img.convertTo(floatImage, CV_32FC1, 1.0 / 255.0);

      std::vector<int64_t> inputShape{1, 1, floatImage.rows, floatImage.cols};
      Ort::MemoryInfo memoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);

      Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
          memoryInfo,
          reinterpret_cast<float *>(floatImage.data),
          floatImage.total(),
          inputShape.data(),
          inputShape.size());

      OPENMVG_LOG_INFO << "Input tensor prepared with shape: [" << floatImage.rows << ", " << floatImage.cols << "]";

      const char *input_names[] = {"image"};
      const char *output_names[] = {"keypoints", "scores", "descriptors"};
      auto outputs = this->session_->Run(Ort::RunOptions{nullptr}, input_names, &inputTensor, 1, output_names, 3);

      OPENMVG_LOG_INFO << "Inference completed successfully";

      const auto &kp_output = outputs[0];    // keypoints [1, N, 2]
      const auto &score_output = outputs[1]; // scores [1, N]
      const auto &desc_output = outputs[2];  // descriptors [1, 256, N]

      const float *kp_data = kp_output.GetTensorData<float>();
      const float *score_data = score_output.GetTensorData<float>();
      const float *desc_data = desc_output.GetTensorData<float>();

      const int num_keypoints = kp_output.GetTensorTypeAndShapeInfo().GetShape()[1];

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
        const float x = kp_data[i * 2 + 0];
        const float y = kp_data[i * 2 + 1];
        regions->Features().emplace_back(x, y, 0.0f, 0.0f);

        openMVG::features::Scalar_Regions<openMVG::features::SIOPointFeature, float, 256>::DescriptorT descriptor;
        const float *desc_start = desc_data + i * 256;
        std::copy(desc_start, desc_start + 256, descriptor.data());
        regions->Descriptors().push_back(descriptor);
      }

      OPENMVG_LOG_INFO << "Feature extraction completed";
      return regions;
    }
    catch (const Ort::Exception &e)
    {
      OPENMVG_LOG_ERROR << "ONNX Runtime error: " << e.what();
    }
    catch (const cv::Exception &e)
    {
      OPENMVG_LOG_ERROR << "OpenCV error: " << e.what();
    }
    catch (const std::exception &e)
    {
      OPENMVG_LOG_ERROR << "Standard exception: " << e.what();
    }
    catch (...)
    {
      OPENMVG_LOG_ERROR << "Unknown error occurred in Describe";
    }

    return std::make_unique<openMVG::features::SuperPoint_Regions>();
  }

  std::unique_ptr<openMVG::features::Regions> Allocate() const override
  {
    return std::unique_ptr<openMVG::features::SuperPoint_Regions>(new openMVG::features::SuperPoint_Regions);
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
  // std::string sImage_Describer_Method = "SUPERPOINT";
  std::string sImage_Describer_Method = "SIFT";
  bool bForce = false;
  std::string sFeaturePreset = "";
#ifdef OPENMVG_USE_OPENMP
  int iNumThreads = 0;
#endif

  // required
  cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
  cmd.add(make_option('o', sOutDir, "outdir"));
  // Optional
  cmd.add(make_option('m', sImage_Describer_Method, "describerMethod"));
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
        << "[-m|--describerMethod]\n"
        << "  (method to use to describe an image):\n"
        << "   SIFT (default),\n"
        << "   SIFT_ANATOMY,\n"
        << "   AKAZE_FLOAT: AKAZE with floating point descriptors,\n"
        << "   AKAZE_MLDB:  AKAZE with binary descriptors\n"
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

  sImage_Describer_Method = "SUPERPOINT";

  OPENMVG_LOG_INFO
      << " You called : " << "\n"
      << argv[0] << "\n"
      << "--input_file " << sSfM_Data_Filename << "\n"
      << "--outdir " << sOutDir << "\n"
      << "--describerMethod " << sImage_Describer_Method << "\n"
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
  std::unique_ptr<Image_describer> image_describer;

  const std::string sImage_describer = stlplus::create_filespec(sOutDir, "image_describer", "json");

  // if (!bForce && stlplus::is_file(sImage_describer))
  // {

  //   // Dynamically load the image_describer from the file (will restore old used settings)
  //   std::ifstream stream(sImage_describer.c_str());
  //   if (!stream)
  //     return EXIT_FAILURE;

  //   try
  //   {
  //     cereal::JSONInputArchive archive(stream);
  //     archive(cereal::make_nvp("image_describer", image_describer));
  //   }
  //   catch (const cereal::Exception &e)
  //   {
  //     OPENMVG_LOG_ERROR << e.what() << '\n'
  //                       << "Cannot dynamically allocate the Image_describer interface.";
  //     return EXIT_FAILURE;
  //   }
  // }
  // else
  {
    // Create the desired Image_describer method.
    // Don't use a factory, perform direct allocation
    if (sImage_Describer_Method == "SIFT")
    {
      image_describer.reset(new SIFT_Image_describer(SIFT_Image_describer::Params(), !bUpRight));
    }
    else if (sImage_Describer_Method == "SIFT_ANATOMY")
    {
      image_describer.reset(
          new SIFT_Anatomy_Image_describer(SIFT_Anatomy_Image_describer::Params()));
    }
    else if (sImage_Describer_Method == "AKAZE_FLOAT")
    {
      image_describer = AKAZE_Image_describer::create(AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MSURF), !bUpRight);
    }
    else if (sImage_Describer_Method == "AKAZE_MLDB")
    {
      image_describer = AKAZE_Image_describer::create(AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MLDB), !bUpRight);
    }
    else if (sImage_Describer_Method == "SUPERPOINT")
    {
      image_describer.reset(new SuperPoint_Image_describer("/model/superpoint.onnx"));
    }
    if (!image_describer)
    {
      OPENMVG_LOG_ERROR << "Cannot create the designed Image_describer:"
                        << sImage_Describer_Method << ".";
      return EXIT_FAILURE;
    }
    else
    {
      if (!sFeaturePreset.empty())
        if (!image_describer->Set_configuration_preset(stringToEnum(sFeaturePreset)))
        {
          OPENMVG_LOG_ERROR << "Preset configuration failed.";
          return EXIT_FAILURE;
        }
    }

    // Export the used Image_describer and region type for:
    // - dynamic future regions computation and/or loading
    // {
    //   std::ofstream stream(sImage_describer.c_str());
    //   if (!stream)
    //     return EXIT_FAILURE;

    //   cereal::JSONOutputArchive archive(stream);
    //   archive(cereal::make_nvp("image_describer", image_describer));
    //   auto regionsType = image_describer->Allocate();
    //   archive(cereal::make_nvp("regions_type", regionsType));
    // }
  }

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
#ifdef OPENMVG_USE_OPENMP
    const unsigned int nb_max_thread = omp_get_max_threads();

    if (iNumThreads > 0)
    {
      omp_set_num_threads(iNumThreads);
    }
    else
    {
      omp_set_num_threads(nb_max_thread);
    }

#pragma omp parallel for schedule(dynamic) private(imageGray)
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
        if (!ReadImage(sView_filename.c_str(), &imageGray))
          continue;

        //
        // Look if there is an occlusion feature mask
        //
        openMVG::image::Image<unsigned char> *mask = nullptr; // The mask is null by default

        const std::string
            mask_filename_local =
                stlplus::create_filespec(sfm_data.s_root_path,
                                         stlplus::basename_part(sView_filename) + "_mask", "png"),
            mask_filename_global =
                stlplus::create_filespec(sfm_data.s_root_path, "mask", "png");

        openMVG::image::Image<unsigned char> imageMask;
        // Try to read the local mask
        if (stlplus::file_exists(mask_filename_local))
        {
          if (!ReadImage(mask_filename_local.c_str(), &imageMask))
          {
            OPENMVG_LOG_ERROR
                << "Invalid mask: " << mask_filename_local << ';'
                << "Stopping feature extraction.";
            preemptive_exit = true;
            continue;
          }
          // Use the local mask only if it fits the current image size
          if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
            mask = &imageMask;
        }
        else
        {
          // Try to read the global mask
          if (stlplus::file_exists(mask_filename_global))
          {
            if (!ReadImage(mask_filename_global.c_str(), &imageMask))
            {
              OPENMVG_LOG_ERROR
                  << "Invalid mask: " << mask_filename_global << ';'
                  << "Stopping feature extraction.";
              preemptive_exit = true;
              continue;
            }
            // Use the global mask only if it fits the current image size
            if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
              mask = &imageMask;
          }
        }

        // Compute features and descriptors and export them to files
        auto regions = image_describer->Describe(imageGray, mask);
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
    OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();
  }
  return EXIT_SUCCESS;
}
