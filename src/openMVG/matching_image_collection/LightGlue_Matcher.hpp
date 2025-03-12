

#ifndef OPENMVG_MATCHING_IMAGE_LIGHTGLUE_HPP
#define OPENMVG_MATCHING_IMAGE_LIGHTGLUE_HPP

#include "openMVG/features/regions.hpp"

#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/system/onnxruntime.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching_image_collection/Matcher.hpp"

#include <vector>
#include <memory>
#include <string>

namespace openMVG
{
  using namespace matching;
  namespace matching_image_collection
  {
    class RegionsMatcherLightGlue
    {
    private:
      const features::Regions *regions;
      ONNXRuntime::InferEnv *infer_env;
      std::vector<float> kp0, desc0;

    public:
      RegionsMatcherLightGlue(const features::Regions &_regions, ONNXRuntime::InferEnv *_infer_env) : regions(&_regions), infer_env(_infer_env)
      {
        if (regions->RegionCount() == 0)
        {
          return;
        }
        const float *dataset = static_cast<const float *>(regions->DescriptorRawData());
        int nbRows = regions->RegionCount(), dimension = regions->DescriptorLength();
        kp0.reserve(nbRows * 2);
        desc0.reserve(nbRows * (dimension - 2));

        for (int i = 0; i < nbRows; i++)
        {
          kp0.push_back(dataset[i * 258]);
          kp0.push_back(dataset[i * 258 + 1]);
          desc0.insert(desc0.end(), dataset + i * 258 + 2, dataset + (i + 1) * 258);
        }
        infer_env->set_input("kpts0", kp0, {1, nbRows, 2});
        infer_env->set_input("desc0", desc0, {1, nbRows, 256});
      }

      bool Match(const float threshold, const features::Regions &query_regions, matching::IndMatches &matches)
      {

        if (!regions)
        {
          return false;
        }
        const float *query = reinterpret_cast<const float *>(query_regions.DescriptorRawData());
        const int nbQuery = query_regions.RegionCount();
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

        std::vector<Ort::Value> res = infer_env->infer();

        const Ort::Value &matches0 = res[infer_env->get_output_index("matches0")],
                         &matches1 = res[infer_env->get_output_index("matches1")],
                         &scores0 = res[infer_env->get_output_index("mscores0")],
                         &scores1 = res[infer_env->get_output_index("mscores1")];

        const size_t match_cnt_0 = matches0.GetTensorTypeAndShapeInfo().GetShape()[1],
                     match_cnt_1 = matches1.GetTensorTypeAndShapeInfo().GetShape()[1];

        const int64_t *m0 = matches0.GetTensorData<int64_t>(), *m1 = matches1.GetTensorData<int64_t>();
        const float *s0 = scores0.GetTensorData<float>(), *s1 = scores1.GetTensorData<float>();
        std::unordered_map<int64_t, int64_t> match_map_1_0, match_map_0_1;
        for (size_t i = 0; i < match_cnt_0; ++i)
        {
          if (match_map_1_0.count(m0[i]) == 0 || s0[match_map_1_0[m0[i]]] < s0[i])
          {
            match_map_1_0[m0[i]] = i;
          }
        }
        for (size_t i = 0; i < match_cnt_1; ++i)
        {
          if (match_map_0_1.count(m1[i]) == 0 || s1[match_map_0_1[m1[i]]] < s1[i])
          {
            match_map_0_1[m1[i]] = i;
          }
        }

        for (const auto &[idx1, idx0] : match_map_1_0)
        {
          auto it = match_map_0_1.find(idx0);
          if (it != match_map_0_1.end() && it->second == idx1)
          {
            const float score = 0.5f * (s0[idx0] + s1[idx1]);
            if (score >= threshold)
            {
              matches.emplace_back(idx1, idx0);
            }
          }
        }
        return true;
      };
    };

    class LightGlue_Matcher_Regions : public Matcher
    {
    private:
      std::unique_ptr<ONNXRuntime::InferEnv> infer_env;
      float score_threshold;

    public:
      LightGlue_Matcher_Regions(float threshold, const char *model_path = "/models/lightglue.onnx") : Matcher(), score_threshold(threshold)
      {
        infer_env.reset(new ONNXRuntime::InferEnv("ONNX LightGlue", model_path));
      }

      void Match(
          const std::shared_ptr<sfm::Regions_Provider> &regions_provider,
          const Pair_Set &pairs,
          PairWiseMatchesContainer &map_PutativeMatches,
          system::ProgressInterface *my_progress_bar = nullptr) const override
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
            matcher.Match(score_threshold, *regionsJ.get(), vec_putative_matches);

            if (!vec_putative_matches.empty())
            {
              map_PutativeMatches.insert({{I, J}, std::move(vec_putative_matches)});
            }

            ++(*my_progress_bar);
          }
        }
      }
    };
  }
}
#endif
