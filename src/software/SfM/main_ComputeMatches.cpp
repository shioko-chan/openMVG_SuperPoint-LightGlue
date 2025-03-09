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

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

using namespace openMVG;
using namespace openMVG::matching;
using namespace openMVG::sfm;
using namespace openMVG::matching_image_collection;

class InferEnv
{
private:
  std::unique_ptr<Ort::Session> session;
  std::vector<Ort::Value> inputs;
  Ort::MemoryInfo memory_info{nullptr};
  std::vector<std::string> input_names, output_names;
  std::vector<const char *> input_names_cstr, output_names_cstr;

public:
  InferEnv() = default;
  InferEnv(const char *name, const char *model_path, const OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_INFO)
  {
    Ort::Env env(log_level, name);

    Ort::SessionOptions session_options;

    auto providers = Ort::GetAvailableProviders();
    std::string available_providers;
    for (auto &&provider : providers)
    {
      available_providers += provider + " ";
    }
    OPENMVG_LOG_INFO << "Available providers are: [" << available_providers << "]";

    // OrtTensorRTProviderOptions provider_options;
    // provider_options.device_id = 0;
    // provider_options.trt_engine_cache_path = "/models";
    // provider_options.trt_engine_cache_enable = 1;
    // provider_options.trt_max_workspace_size = 4 * (1UL << 30); // 4GB
    // provider_options.trt_fp16_enable = 1;
    // OPENMVG_LOG_INFO << "TensorRT PROVIDER";
    // try
    // {
    // session_options.AppendExecutionProvider_TensorRT(provider_options);
    // }
    // catch (const std::exception &e)
    // {
    //   OPENMVG_LOG_ERROR << "Error loading TensorRT provider: " << e.what();
    //   throw;
    // }
    // OPENMVG_LOG_INFO << "TensorRT PROVIDER APPENDED";

    // OPENMVG_LOG_INFO << "TensorRT Execution Provider enabled";
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    // session_options.SetOptimizedModelFilePath(model_path);

    OPENMVG_LOG_INFO << "Model path: " << model_path;
    try
    {
      session.reset(new Ort::Session(env, model_path, session_options));
    }
    catch (const std::exception &e)
    {
      OPENMVG_LOG_ERROR << "Error loading model: " << e.what();
      throw;
    }
    OPENMVG_LOG_INFO << "Session created";
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < session->GetInputCount(); ++i)
    {
      input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
      inputs.push_back(Ort::Value(nullptr));
    }
    for (int i = 0; i < session->GetOutputCount(); ++i)
    {
      output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    }
    std::transform(input_names.begin(), input_names.end(), std::back_inserter(input_names_cstr), [](std::string const &s)
                   { return s.c_str(); });
    std::transform(output_names.begin(), output_names.end(), std::back_inserter(output_names_cstr), [](std::string const &s)
                   { return s.c_str(); });

    OPENMVG_LOG_INFO << "Model loaded: " << model_path;
    memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  }
  template <typename T>
  void set_input(std::string const &name, std::vector<T> &input, std::vector<int64_t> const &shape)
  {
    size_t idx = std::find(input_names.begin(), input_names.end(), name) - input_names.begin();
    if (idx >= input_names.size())
    {
      throw std::runtime_error("Input name not found");
    }
    OPENMVG_LOG_INFO << "Setting input: " << name << " idx: " << idx;
    inputs[idx] = Ort::Value::CreateTensor<T>(memory_info, input.data(), input.size(), shape.data(), shape.size());
  }

  const std::vector<Ort::Value> infer()
  {
    return session->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), inputs.data(), input_names.size(), output_names_cstr.data(), output_names.size());
  }

  const std::vector<std::string> &get_input_names() const
  {
    return input_names;
  }

  const std::vector<std::string> &get_output_names() const
  {
    return output_names;
  }
};

class RegionsMatcherLightGlue
{
private:
  const features::Regions *regions;
  InferEnv *infer_env;
  std::vector<float> kp0, desc0;

public:
  RegionsMatcherLightGlue(const features::Regions &_regions, InferEnv *_infer_env) : regions(&_regions), infer_env(_infer_env)
  {
    if (regions->RegionCount() == 0)
    {
      return;
    }
    const float *dataset = static_cast<const float *>(regions->DescriptorRawData());
    int nbRows = regions->RegionCount(), dimension = regions->DescriptorLength();
    kp0.reserve(nbRows * 2);
    desc0.reserve(nbRows * (dimension - 2));
    OPENMVG_LOG_INFO << "dimension:" << dimension << " nbRows:" << nbRows;
    assert(("not a case", dimension == 258));
    for (int i = 0; i < nbRows; i++)
    {
      kp0.push_back(dataset[i * 258]);
      kp0.push_back(dataset[i * 258 + 1]);
      desc0.insert(desc0.end(), dataset + i * 258 + 2, dataset + (i + 1) * 258);
    }
    infer_env->set_input("kpts0", kp0, {1, nbRows, 2});
    infer_env->set_input("desc0", desc0, {1, nbRows, 256});
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
  bool SearchNeighbours(const float *query, int nbQuery, IndMatches *indices, std::vector<float> *distances, size_t NN)
  {

    OPENMVG_LOG_INFO << "Searching for " << nbQuery << " queries";

    std::vector<float> kp1, desc1;
    kp1.reserve(nbQuery * 2);
    desc1.reserve(nbQuery * 256);

    for (int i = 0; i < nbQuery; i++)
    {
      kp1.push_back(query[i * 258]);
      kp1.push_back(query[i * 258 + 1]);
      desc1.insert(desc1.end(), query + i * 258 + 2, query + (i + 1) * 258);
    }

    infer_env->set_input("kpts1", kp1, {1, nbQuery, 2});
    infer_env->set_input("desc1", desc1, {1, nbQuery, 256});

    auto res = infer_env->infer();

    for (auto &r : res)
    {
      auto shape = r.GetTensorTypeAndShapeInfo().GetShape();
      std::string shape_str;
      for (int i = 0; i < shape.size(); i++)
      {
        shape_str += std::to_string(shape[i]) + " ";
      }
      OPENMVG_LOG_INFO << "Output shape: " << shape_str;
      // auto data = r.GetTensorMutableData<float>();
    }
    return true;
  };

  bool MatchDistanceRatio(
      const float distance_ratio,
      const features::Regions &query_regions,
      matching::IndMatches &matches)
  {
    if (!regions)
      return false;

    const float *queries = reinterpret_cast<const float *>(query_regions.DescriptorRawData());

    const size_t number_neighbor = 2;
    matching::IndMatches nn_matches;
    std::vector<float> nn_distances;

    // Search the 2 closest neighbours for each query descriptor
    if (!this->SearchNeighbours(queries,
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
        Square(distance_ratio));

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
  std::unique_ptr<InferEnv> infer_env;
  float f_dist_ratio_;

public:
  LightGlue_Matcher_Regions(float distRatio, const char *model_path = "/models/lightglue.onnx") : Matcher(), f_dist_ratio_(distRatio)
  {
    infer_env.reset(new InferEnv("ONNX LightGlue", model_path));
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

      RegionsMatcherLightGlue matcher(*regionsI.get(), infer_env.get());

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
        matcher.MatchDistanceRatio(f_dist_ratio_, *regionsJ.get(), vec_putative_matches);

        if (!vec_putative_matches.empty())
        {
          map_PutativeMatches.insert({{I, J}, std::move(vec_putative_matches)});
        }

        ++(*my_progress_bar);
      }
    }
  }
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

  //!!---------------------------------------
  sNearestMatchingMethod = "LIGHTGLUE";
  //!!---------------------------------------
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
  //!!--------------------------------------------------------------
  regions_type.reset(new features::SuperPoint_Regions());
  //!!--------------------------------------------------------------
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
