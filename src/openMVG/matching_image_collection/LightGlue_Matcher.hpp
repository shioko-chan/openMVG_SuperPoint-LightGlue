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
#include <unordered_set>
#include <memory>
#include <string>

namespace openMVG
{
  using namespace matching;
  using namespace features;
  using namespace ONNXRuntime;
  namespace matching_image_collection
  {
    class RegionsMatcherLightGlue
    {
    private:
      const Regions *regions;
      InferEnv *infer_env;
      std::vector<float> kp0, desc0, kp1, desc1;
      float threshold;

    public:
      RegionsMatcherLightGlue(const float threshold, const Regions &_regions, InferEnv *_infer_env) : threshold(threshold), regions(&_regions), infer_env(_infer_env)
      {
        if (regions->RegionCount() == 0)
        {
          return;
        }
        const float *dataset = static_cast<const float *>(regions->DescriptorRawData());
        int nbRows = regions->RegionCount(), dimension = regions->DescriptorLength();

        if (dimension != 258)
        {
          throw std::runtime_error("LightGlue only supports 258-dimensional descriptors");
        }

        kp0.reserve(nbRows * 2);
        desc0.reserve(nbRows * 256);

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
       * \param[in]   query_regions     The query array.
       * \param[out]  matches   The corresponding (query, neighbor) indices.
       *
       * \return True if success.
       */
      bool Match(const Regions &query_regions, IndMatches &matches)
      {

        if (!regions)
        {
          return false;
        }
        const float *query = reinterpret_cast<const float *>(query_regions.DescriptorRawData());
        const int nbQuery = query_regions.RegionCount();

        kp1.resize(0);
        desc1.resize(0);

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

        std::unordered_set<std::pair<int64_t, int64_t>> matches_set;
        for (int64_t i = 0; i < match_cnt_0; ++i)
        {
          if (m0[i] >= 0 && m1[m0[i]] == i && s0[i] >= threshold)
          {
            matches_set.emplace(i, m0[i]);
          }
        }
        for (int64_t i = 0; i < match_cnt_1; ++i)
        {
          if (m1[i] >= 0 && m0[m1[i]] == i && s1[i] >= threshold)
          {
            matches_set.emplace(m1[i], i);
          }
        }
        std::transform(matches_set.begin(), matches_set.end(), std::back_inserter(matches), [](const std::pair<int64_t, int64_t> &p)
                       { return IndMatch(p.first, p.second); });

        return true;
      };
    };

    class LightGlue_Matcher_Regions : public Matcher
    {
    private:
      std::unique_ptr<InferEnv> infer_env;
      float threshold;

    public:
      LightGlue_Matcher_Regions(float threshold, const char *model_path = "/models/lightglue.onnx") : Matcher(), threshold(threshold)
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
        {
          my_progress_bar = &system::ProgressInterface::dummy();
        }
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
          {
            continue;
          }
          const IndexT I = pairs_it.first;
          const auto &indexToCompare = pairs_it.second;

          const std::shared_ptr<Regions> regionsI = regions_provider->get(I);
          if (regionsI->RegionCount() == 0)
          {
            (*my_progress_bar) += indexToCompare.size();
            continue;
          }

          RegionsMatcherLightGlue matcher(threshold, *regionsI.get(), infer_env.get());

          for (int j = 0; j < static_cast<int>(indexToCompare.size()); ++j)
          {
            const IndexT J = indexToCompare[j];

            const std::shared_ptr<Regions> regionsJ = regions_provider->get(J);
            if (regionsJ->RegionCount() == 0 || regionsI->Type_id() != regionsJ->Type_id())
            {
              ++(*my_progress_bar);
              continue;
            }

            IndMatches vec_putative_matches;
            matcher.Match(*regionsJ.get(), vec_putative_matches);

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
