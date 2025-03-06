// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2019 Pierre MOULON, Romuald Perrot.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/graph/graph.hpp"
#include "openMVG/graph/graph_stats.hpp"
#include "openMVG/matching/matching_interface.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching/pairwiseAdjacencyDisplay.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Pair_Builder.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_preemptive_regions_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/timer.hpp"

#include "openMVG/tensorrt.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace openMVG;
using namespace openMVG::matching;
using namespace openMVG::sfm;
using namespace openMVG::matching_image_collection;

#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

using namespace openMVG::TensorRT;

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

  const size_t min_kp = 20, avg_kp = 3000, max_kp = 6000;
  bool set_dimensions(nvinfer1::IOptimizationProfile *profile, const char *name, const size_t n)
  {
    return profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, min_kp, n)) &&
           profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, avg_kp, n)) &&
           profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, max_kp, n));
  }

public:
  NVInferEnv(const std::string &model_path = "/models/lightglue.onnx") : logger(nvinfer1::ILogger::Severity::kINFO)
  {
    OPENMVG_LOG_INFO << "Creating TensorRT environment";

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

    if (!set_dimensions(profile, "kpts0", 2) ||
        !set_dimensions(profile, "kpts1", 2) ||
        !set_dimensions(profile, "desc0", 256) ||
        !set_dimensions(profile, "desc1", 256))
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
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4 * (static_cast<size_t>(1) << 30)); // 4GB

    this->engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  }

  std::unique_ptr<nvinfer1::IExecutionContext> create_context()
  {
    return std::unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
  };
};

// By default compute square(L2 distance).
template <typename Scalar = float, typename Metric = L2<Scalar>>
class ArrayMatcherLightGlue : public ArrayMatcher<float, L2<float>>
{
private:
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream;
  HostAllocator kp0_h, kp1_h, de0_h, de1_h, match0_h, match1_h, score0_h, score1_h;
  GPUAllocator kp0, kp1, de0, de1;
  OutputAllocator match0, match1, score0, score1;

public:
  using DistanceType = typename Metric::ResultType;

  void init_context(std::unique_ptr<nvinfer1::IExecutionContext> ctx)
  {
    context = std::move(ctx);
    cudaStreamCreate(&stream);
    kp0 = GPUAllocator(stream);
    kp1 = GPUAllocator(stream);
    de0 = GPUAllocator(stream);
    de1 = GPUAllocator(stream);
    match0 = OutputAllocator(stream);
    match1 = OutputAllocator(stream);
    score0 = OutputAllocator(stream);
    score1 = OutputAllocator(stream);

    this->context->setOptimizationProfileAsync(0, this->stream);

    this->context->setOutputAllocator("matches0", &match0);
    this->context->setOutputAllocator("matches1", &match1);
    this->context->setOutputAllocator("mscores0", &score0);
    this->context->setOutputAllocator("mscores1", &score1);

    this->context->setOutputTensorAddress("matches0", nullptr);
    this->context->setOutputTensorAddress("matches1", nullptr);
    this->context->setOutputTensorAddress("mscores0", nullptr);
    this->context->setOutputTensorAddress("mscores1", nullptr);
  }

  ArrayMatcherLightGlue() = default;
  ~ArrayMatcherLightGlue()
  {
    cudaStreamDestroy(stream);
  }

  // HostAllocator kp0_h, kp1_h, de0_h, de1_h, match0_h, match1_h, score0_h, score1_h;
  // GPUAllocator kp0, kp1, de0, de1;
  // OutputAllocator match0, match1, score0, score1;
  /**
   * Build the matching structure
   *
   * \param[in] dataset   Input data.
   * \param[in] nbRows    The number of component.
   * \param[in] dimension Length of the data contained in the dataset.
   *
   * \return True if success.
   */
  bool Build(const Scalar *dataset, int nbRows, int dimension) override
  {
    assert(dimension == 258);
    // if engine
    kp0.reallocate(nbRows * 2 * sizeof(float));
    de0.reallocate(nbRows * 256 * sizeof(float));
    kp0_h.reallocate(nbRows * 2 * sizeof(float));
    de0_h.reallocate(nbRows * 256 * sizeof(float));

    for (int i = 0; i < nbRows; i++)
    {
      static_cast<float *>(kp0_h.ptr)[i * 2] = dataset[i * 258];
      static_cast<float *>(kp0_h.ptr)[i * 2 + 1] = dataset[i * 258 + 1];
      std::memcpy(static_cast<void *>(static_cast<float *>(de0_h.ptr) + i * 256), static_cast<const void *>(dataset + i * 258 + 2), 256 * sizeof(float));
    }

    cudaMemcpyAsync(kp0.ptr, kp0_h.ptr, nbRows * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(de0.ptr, de0_h.ptr, nbRows * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);

    this->context->setInputShape("kpts0", nvinfer1::Dims3(1, nbRows, 2));
    this->context->setInputShape("desc0", nvinfer1::Dims3(1, nbRows, 256));

    this->context->setInputTensorAddress("kpts0", kp0.ptr);
    this->context->setInputTensorAddress("desc0", de0.ptr);
    return true;
  };

  bool SearchNeighbour(
      const Scalar *query,
      int *indice,
      DistanceType *distance) override
  {
    OPENMVG_LOG_ERROR << "NOT IMPLEMENTED !!!";
    return false;
  }

  /**
   * Search the N nearest Neighbor of the scalar array query.
   *
   * \param[in]   query     The query array.
   * \param[in]   nbQuery   The number of query rows.
   * \param[out]  indices   The corresponding (query, neighbor) indices.
   * \param[out]  distances The distances between the matched arrays.
   * \param[in]  NN        The number of maximal neighbor that will be searched.
   *
   * \return True if success.
   */
  bool SearchNeighbours(
      const Scalar *query, int nbQuery,
      IndMatches *indices,
      std::vector<DistanceType> *distances,
      size_t NN) override
  {
    kp1_h.reallocate(nbQuery * 2 * sizeof(float));
    de1_h.reallocate(nbQuery * 256 * sizeof(float));
    kp1.reallocate(nbQuery * 2 * sizeof(float));
    de1.reallocate(nbQuery * 256 * sizeof(float));

    for (int i = 0; i < nbQuery; i++)
    {
      static_cast<float *>(kp1_h.ptr)[i * 2] = query[i * 258];
      static_cast<float *>(kp1_h.ptr)[i * 2 + 1] = query[i * 258 + 1];
      std::memcpy(static_cast<void *>(static_cast<float *>(de1_h.ptr) + i * 256), static_cast<const void *>(query + i * 258 + 2), 256 * sizeof(float));
    }

    cudaMemcpyAsync(kp1.ptr, kp1_h.ptr, nbQuery * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(de1.ptr, de1_h.ptr, nbQuery * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);

    this->context->setInputShape("kpts1", nvinfer1::Dims3(1, nbQuery, 2));
    this->context->setInputShape("desc1", nvinfer1::Dims3(1, nbQuery, 256));

    this->context->setInputTensorAddress("kpts1", kp1.ptr);
    this->context->setInputTensorAddress("desc1", de1.ptr);

    if (!this->context->enqueueV3(stream))
    {
      OPENMVG_LOG_ERROR << "Failed to enqueue inference";
      return false;
    }

    cudaStreamSynchronize(this->stream);

    print_dims(this->match0.output_dims, "matches0");
    print_dims(this->match1.output_dims, "matches1");
    print_dims(this->score0.output_dims, "scores0");
    print_dims(this->score1.output_dims, "scores1");
    return true;
  };

  const char *print_dims(nvinfer1::Dims &dims, const char *name)
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
};

template <class ArrayMatcherT>
class RegionsMatcherLightGlue : public RegionsMatcher
{
private:
  ArrayMatcherT matcher_;
  const features::Regions *regions_;
  const bool b_squared_metric_; // Store if the metric is squared or not
public:
  using Scalar = typename ArrayMatcherT::ScalarT;
  using DistanceType = typename ArrayMatcherT::DistanceType;

  RegionsMatcherLightGlue() : regions_(nullptr), b_squared_metric_(false)
  {
  }

  /**
   * @brief Init the matcher with some reference regions.
   */
  RegionsMatcherLightGlue(
      const features::Regions &regions,
      std::unique_ptr<nvinfer1::IExecutionContext> context,
      bool b_squared_metric = false) : regions_(&regions),
                                       b_squared_metric_(b_squared_metric)
  {
    if (regions_->RegionCount() == 0)
      return;
    matcher_.init_context(std::move(context));
    const Scalar *tab = reinterpret_cast<const Scalar *>(regions_->DescriptorRawData());
    matcher_.Build(tab, regions_->RegionCount(), regions_->DescriptorLength());
  }

  bool Match(
      const features::Regions &query_regions,
      matching::IndMatches &matches) override
  {
    if (!regions_)
      return false;

    const Scalar *queries = reinterpret_cast<const Scalar *>(query_regions.DescriptorRawData());

    // Search the closest neighbour for each query descriptor
    std::vector<DistanceType> distances;
    matches.clear();
    if (!matcher_.SearchNeighbours(queries, query_regions.RegionCount(), &matches, &distances, 1))
      return false;

    for (auto &match : matches)
    {
      std::swap(match.i_, match.j_);
    }

    return (!matches.empty());
  }

  /**
   * @brief Match some regions to the database of internal regions.
   */
  bool MatchDistanceRatio(
      const float distance_ratio,
      const features::Regions &query_regions,
      matching::IndMatches &matches) override
  {
    if (!regions_)
      return false;

    const Scalar *queries = reinterpret_cast<const Scalar *>(query_regions.DescriptorRawData());

    const size_t number_neighbor = 2;
    matching::IndMatches nn_matches;
    std::vector<DistanceType> nn_distances;

    // Search the 2 closest neighbours for each query descriptor
    if (!matcher_.SearchNeighbours(queries,
                                   query_regions.RegionCount(),
                                   &nn_matches,
                                   &nn_distances,
                                   number_neighbor))
      return false;

    std::vector<int> nn_ratio_indexes;
    // Filter the matches using a distance ratio test:
    //   The probability that a match is correct is determined by taking
    //   the ratio of distance from the closest neighbor to the distance
    //   of the second closest.
    matching::NNdistanceRatio(
        nn_distances.cbegin(), // distance start
        nn_distances.cend(),   // distance end
        number_neighbor,       // Number of neighbor in iterator sequence (minimum required 2)
        nn_ratio_indexes,      // output (indices that respect the distance Ratio)
        b_squared_metric_ ? Square(distance_ratio) : distance_ratio);

    matches.clear();
    matches.reserve(nn_ratio_indexes.size());
    for (const auto &index : nn_ratio_indexes)
    {
      matches.emplace_back(nn_matches[index * number_neighbor].j_,
                           nn_matches[index * number_neighbor].i_);
    }

    return (!matches.empty());
  }
};

class LightGlue_Matcher_Regions : public Matcher
{
private:
  std::unique_ptr<NVInferEnv> env;

public:
  LightGlue_Matcher_Regions(float distRatio) : Matcher(), f_dist_ratio_(distRatio)
  {
    env.reset(new NVInferEnv("/models/lightglue.onnx"));
  }

  void Match(
      const std::shared_ptr<sfm::Regions_Provider> &regions_provider,
      const Pair_Set &pairs,
      PairWiseMatchesContainer &map_PutativeMatches,
      system::ProgressInterface *my_progress_bar) const override
  {
    if (!my_progress_bar)
      my_progress_bar = &system::ProgressInterface::dummy();

    // #ifdef OPENMVG_USE_OPENMP
    //     OPENMVG_LOG_INFO << "Using the OPENMP thread interface";
    // #endif

    my_progress_bar->Restart(pairs.size(), "- Matching -");

    // Sort pairs according the first index to minimize the MatcherT build operations
    using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
    Map_vectorT map_Pairs;
    for (const auto &pair_it : pairs)
    {
      map_Pairs[pair_it.first].push_back(pair_it.second);
    }

    // Perform matching between all the pairs
    for (const auto &pairs_it : map_Pairs)
    {
      if (my_progress_bar->hasBeenCanceled())
        continue;
      const IndexT I = pairs_it.first;
      const auto &indexToCompare = pairs_it.second;

      const std::shared_ptr<features::Regions> regionsI = regions_provider->get(I);
      if (regionsI->RegionCount() == 0)
      {
        (*my_progress_bar) += indexToCompare.size();
        continue;
      }

      // Initialize the matching interface
      using CurrentMatcherType = RegionsMatcherLightGlue<ArrayMatcherLightGlue<float, L2<float>>>;
      auto matcher = std::unique_ptr<CurrentMatcherType>(new CurrentMatcherType(*regionsI.get(), env->create_context(), true));

      // #ifdef OPENMVG_USE_OPENMP
      // #pragma omp parallel for schedule(dynamic)
      // #endif
      for (int j = 0; j < static_cast<int>(indexToCompare.size()); ++j)
      {
        const IndexT J = indexToCompare[j];

        const std::shared_ptr<features::Regions> regionsJ = regions_provider->get(J);
        if (regionsJ->RegionCount() == 0 || regionsI->Type_id() != regionsJ->Type_id())
        {
          ++(*my_progress_bar);
          continue;
        }

        IndMatches vec_putative_matches;
        matcher->MatchDistanceRatio(f_dist_ratio_, *regionsJ.get(), vec_putative_matches);

        // #ifdef OPENMVG_USE_OPENMP
        // #pragma omp critical
        // #endif
        {
          if (!vec_putative_matches.empty())
          {
            map_PutativeMatches.insert({{I, J}, std::move(vec_putative_matches)});
          }
        }
        ++(*my_progress_bar);
      }
    }
  }

private:
  float f_dist_ratio_;
};

/// Compute corresponding features between a series of views:
/// - Load view images description (regions: features & descriptors)
/// - Compute putative local feature matches (descriptors matching)
int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutputMatchesFilename = "";
  float fDistRatio = 0.8f;
  std::string sPredefinedPairList = "";
  std::string sNearestMatchingMethod = "AUTO";
  bool bForce = false;
  unsigned int ui_max_cache_size = 0;

  // Pre-emptive matching parameters
  unsigned int ui_preemptive_feature_count = 200;
  double preemptive_matching_percentage_threshold = 0.08;

  // required
  cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
  cmd.add(make_option('o', sOutputMatchesFilename, "output_file"));
  cmd.add(make_option('p', sPredefinedPairList, "pair_list"));
  // Options
  cmd.add(make_option('r', fDistRatio, "ratio"));
  cmd.add(make_option('n', sNearestMatchingMethod, "nearest_matching_method"));
  cmd.add(make_option('f', bForce, "force"));
  cmd.add(make_option('c', ui_max_cache_size, "cache_size"));
  // Pre-emptive matching
  cmd.add(make_option('P', ui_preemptive_feature_count, "preemptive_feature_count"));

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
        << "[-i|--input_file]   A SfM_Data file\n"
        << "[-o|--output_file]  Output file where computed matches are stored\n"
        << "[-p|--pair_list]    Pairs list file\n"
        << "\n[Optional]\n"
        << "[-f|--force] Force to recompute data]\n"
        << "[-r|--ratio] Distance ratio to discard non meaningful matches\n"
        << "   0.8: (default).\n"
        << "[-n|--nearest_matching_method]\n"
        << "  AUTO: auto choice from regions type,\n"
        << "  For Scalar based regions descriptor:\n"
        << "    BRUTEFORCEL2: L2 BruteForce matching,\n"
        << "    HNSWL2: L2 Approximate Matching with Hierarchical Navigable Small World graphs,\n"
        << "    HNSWL1: L1 Approximate Matching with Hierarchical Navigable Small World graphs\n"
        << "      tailored for quantized and histogram based descriptors (e.g uint8 RootSIFT)\n"
        << "    CASCADEHASHINGL2: L2 Cascade Hashing matching.\n"
        << "    FASTCASCADEHASHINGL2: (default)\n"
        << "      L2 Cascade Hashing with precomputed hashed regions\n"
        << "     (faster than CASCADEHASHINGL2 but use more memory).\n"
        << "  For Binary based descriptor:\n"
        << "    BRUTEFORCEHAMMING: BruteForce Hamming matching,\n"
        << "    HNSWHAMMING: Hamming Approximate Matching with Hierarchical Navigable Small World graphs\n"
        << "[-c|--cache_size]\n"
        << "  Use a regions cache (only cache_size regions will be stored in memory)\n"
        << "  If not used, all regions will be load in memory."
        << "\n[Pre-emptive matching:]\n"
        << "[-P|--preemptive_feature_count] <NUMBER> Number of feature used for pre-emptive matching";

    OPENMVG_LOG_INFO << s;
    return EXIT_FAILURE;
  }

  // ------
  sNearestMatchingMethod = "LIGHTGLUE";
  // ------

  OPENMVG_LOG_INFO << " You called : "
                   << "\n"
                   << argv[0] << "\n"
                   << "--input_file " << sSfM_Data_Filename << "\n"
                   << "--output_file " << sOutputMatchesFilename << "\n"
                   << "--pair_list " << sPredefinedPairList << "\n"
                   << "Optional parameters:"
                   << "\n"
                   << "--force " << bForce << "\n"
                   << "--ratio " << fDistRatio << "\n"
                   << "--nearest_matching_method " << sNearestMatchingMethod << "\n"
                   << "--cache_size " << ((ui_max_cache_size == 0) ? "unlimited" : std::to_string(ui_max_cache_size)) << "\n"
                   << "--preemptive_feature_used/count " << cmd.used('P') << " / " << ui_preemptive_feature_count;
  if (cmd.used('P'))
  {
    OPENMVG_LOG_INFO << "--preemptive_feature_count " << ui_preemptive_feature_count;
  }

  if (sOutputMatchesFilename.empty())
  {
    OPENMVG_LOG_ERROR << "No output file set.";
    return EXIT_FAILURE;
  }

  // -----------------------------
  // . Load SfM_Data Views & intrinsics data
  // . Compute putative descriptor matches
  // + Export some statistics
  // -----------------------------

  //---------------------------------------
  // Read SfM Scene (image view & intrinsics data)
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS)))
  {
    OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read.";
    return EXIT_FAILURE;
  }
  const std::string sMatchesDirectory = stlplus::folder_part(sOutputMatchesFilename);

  //---------------------------------------
  // Load SfM Scene regions
  //---------------------------------------
  // Init the regions_type from the image describer file (used for image regions extraction)
  using namespace openMVG::features;
  const std::string sImage_describer = stlplus::create_filespec(sMatchesDirectory, "image_describer", "json");
  std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
  if (!regions_type)
  {
    OPENMVG_LOG_ERROR << "Invalid: " << sImage_describer << " regions type file.";
    return EXIT_FAILURE;
  }

  //---------------------------------------
  // a. Compute putative descriptor matches
  //    - Descriptor matching (according user method choice)
  //    - Keep correspondences only if NearestNeighbor ratio is ok
  //---------------------------------------

  // Load the corresponding view regions
  std::shared_ptr<Regions_Provider> regions_provider;
  if (ui_max_cache_size == 0)
  {
    // Default regions provider (load & store all regions in memory)
    regions_provider = std::make_shared<Regions_Provider>();
  }
  else
  {
    // Cached regions provider (load & store regions on demand)
    regions_provider = std::make_shared<Regions_Provider_Cache>(ui_max_cache_size);
  }
  // If we use pre-emptive matching, we load less regions:
  if (ui_preemptive_feature_count > 0 && cmd.used('P'))
  {
    regions_provider = std::make_shared<Preemptive_Regions_Provider>(ui_preemptive_feature_count);
  }

  // Show the progress on the command line:
  system::LoggerProgress progress;

  if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type, &progress))
  {
    OPENMVG_LOG_ERROR << "Cannot load view regions from: " << sMatchesDirectory << ".";
    return EXIT_FAILURE;
  }

  PairWiseMatches map_PutativeMatches;

  // Build some alias from SfM_Data Views data:
  // - List views as a vector of filenames & image sizes
  std::vector<std::string> vec_fileNames;
  std::vector<std::pair<size_t, size_t>> vec_imagesSize;
  {
    vec_fileNames.reserve(sfm_data.GetViews().size());
    vec_imagesSize.reserve(sfm_data.GetViews().size());
    for (const auto view_it : sfm_data.GetViews())
    {
      const View *v = view_it.second.get();
      vec_fileNames.emplace_back(stlplus::create_filespec(sfm_data.s_root_path,
                                                          v->s_Img_path));
      vec_imagesSize.emplace_back(v->ui_width, v->ui_height);
    }
  }

  OPENMVG_LOG_INFO << " - PUTATIVE MATCHES - ";
  // If the matches already exists, reload them
  if (!bForce && (stlplus::file_exists(sOutputMatchesFilename)))
  {
    if (!(Load(map_PutativeMatches, sOutputMatchesFilename)))
    {
      OPENMVG_LOG_ERROR << "Cannot load input matches file";
      return EXIT_FAILURE;
    }
    OPENMVG_LOG_INFO
        << "\t PREVIOUS RESULTS LOADED;"
        << " #pair: " << map_PutativeMatches.size();
  }
  else // Compute the putative matches
  {
    // Allocate the right Matcher according the Matching requested method
    std::unique_ptr<Matcher> collectionMatcher;
    if (sNearestMatchingMethod == "AUTO")
    {
      if (regions_type->IsScalar())
      {
        OPENMVG_LOG_INFO << "Using FAST_CASCADE_HASHING_L2 matcher";
        collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
      }
      else if (regions_type->IsBinary())
      {
        OPENMVG_LOG_INFO << "Using HNSWHAMMING matcher";
        collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_HAMMING));
      }
    }
    else if (sNearestMatchingMethod == "BRUTEFORCEL2")
    {
      OPENMVG_LOG_INFO << "Using BRUTE_FORCE_L2 matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_L2));
    }
    else if (sNearestMatchingMethod == "BRUTEFORCEHAMMING")
    {
      OPENMVG_LOG_INFO << "Using BRUTE_FORCE_HAMMING matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_HAMMING));
    }
    else if (sNearestMatchingMethod == "HNSWL2")
    {
      OPENMVG_LOG_INFO << "Using HNSWL2 matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_L2));
    }
    if (sNearestMatchingMethod == "HNSWL1")
    {
      OPENMVG_LOG_INFO << "Using HNSWL1 matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_L1));
    }
    else if (sNearestMatchingMethod == "HNSWHAMMING")
    {
      OPENMVG_LOG_INFO << "Using HNSWHAMMING matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_HAMMING));
    }
    else if (sNearestMatchingMethod == "CASCADEHASHINGL2")
    {
      OPENMVG_LOG_INFO << "Using CASCADE_HASHING_L2 matcher";
      collectionMatcher.reset(new Matcher_Regions(fDistRatio, CASCADE_HASHING_L2));
    }
    else if (sNearestMatchingMethod == "FASTCASCADEHASHINGL2")
    {
      OPENMVG_LOG_INFO << "Using FAST_CASCADE_HASHING_L2 matcher";
      collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
    }
    else if (sNearestMatchingMethod == "LIGHTGLUE")
    {
      OPENMVG_LOG_INFO << "Using LIGTH_GLUE";
      collectionMatcher.reset(new LightGlue_Matcher_Regions(fDistRatio));
    }
    if (!collectionMatcher)
    {
      OPENMVG_LOG_ERROR << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod;
      return EXIT_FAILURE;
    }
    // Perform the matching
    system::Timer timer;
    {
      // From matching mode compute the pair list that have to be matched:
      Pair_Set pairs;
      if (sPredefinedPairList.empty())
      {
        OPENMVG_LOG_INFO << "No input pair file set. Use exhaustive match by default.";
        const size_t NImage = sfm_data.GetViews().size();
        pairs = exhaustivePairs(NImage);
      }
      else if (!loadPairs(sfm_data.GetViews().size(), sPredefinedPairList, pairs))
      {
        OPENMVG_LOG_ERROR << "Failed to load pairs from file: \"" << sPredefinedPairList << "\"";
        return EXIT_FAILURE;
      }
      OPENMVG_LOG_INFO << "Running matching on #pairs: " << pairs.size();
      // Photometric matching of putative pairs
      collectionMatcher->Match(regions_provider, pairs, map_PutativeMatches, &progress);

      if (cmd.used('P')) // Preemptive filter
      {
        // Keep putative matches only if there is more than X matches
        PairWiseMatches map_filtered_matches;
        for (const auto &pairwisematches_it : map_PutativeMatches)
        {
          const size_t putative_match_count = pairwisematches_it.second.size();
          const int match_count_threshold =
              preemptive_matching_percentage_threshold * ui_preemptive_feature_count;
          // TODO: Add an option to keeping X Best pairs
          if (putative_match_count >= match_count_threshold)
          {
            // the pair will be kept
            map_filtered_matches.insert(pairwisematches_it);
          }
        }
        map_PutativeMatches.clear();
        std::swap(map_filtered_matches, map_PutativeMatches);
      }

      //---------------------------------------
      //-- Export putative matches & pairs
      //---------------------------------------
      if (!Save(map_PutativeMatches, std::string(sOutputMatchesFilename)))
      {
        OPENMVG_LOG_ERROR
            << "Cannot save computed matches in: "
            << sOutputMatchesFilename;
        return EXIT_FAILURE;
      }
      // Save pairs
      const std::string sOutputPairFilename =
          stlplus::create_filespec(sMatchesDirectory, "preemptive_pairs", "txt");
      if (!savePairs(
              sOutputPairFilename,
              getPairs(map_PutativeMatches)))
      {
        OPENMVG_LOG_ERROR
            << "Cannot save computed matches pairs in: "
            << sOutputPairFilename;
        return EXIT_FAILURE;
      }
    }
    OPENMVG_LOG_INFO << "Task (Regions Matching) done in (s): " << timer.elapsed();
  }

  OPENMVG_LOG_INFO << "#Putative pairs: " << map_PutativeMatches.size();

  // -- export Putative View Graph statistics
  graph::getGraphStatistics(sfm_data.GetViews().size(), getPairs(map_PutativeMatches));

  //-- export putative matches Adjacency matrix
  PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
                                       map_PutativeMatches,
                                       stlplus::create_filespec(sMatchesDirectory, "PutativeAdjacencyMatrix", "svg"));
  //-- export view pair graph once putative graph matches has been computed
  {
    std::set<IndexT> set_ViewIds;
    std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(), std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
    graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_PutativeMatches));
    graph::exportToGraphvizData(
        stlplus::create_filespec(sMatchesDirectory, "putative_matches"),
        putativeGraph);
  }

  return EXIT_SUCCESS;
}
