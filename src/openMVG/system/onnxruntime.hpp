#ifndef OPENMVG_ONNXRUNTIME_HPP
#define OPENMVG_ONNXRUNTIME_HPP

#include "openMVG/system/logger.hpp"

#include <variant>
#include <vector>
#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace openMVG
{
  namespace ONNXRuntime
  {

    static Ort::Env &ort_env()
    {
      static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ONNXRUNTIME");
      return env;
    }

    class InferEnv
    {
    private:
      std::unique_ptr<Ort::Session> session;
      std::vector<Ort::Value> inputs;
      std::vector<std::string> input_names, output_names;
      std::vector<const char *> input_names_cstr, output_names_cstr;

    public:
      InferEnv() = delete;
      InferEnv(const char *name, const char *model_path, const OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_ERROR)
      {
        Ort::SessionOptions session_options;

        OrtCUDAProviderOptions provider_options;
        provider_options.device_id = 0;
        provider_options.arena_extend_strategy = 0; // kNextPowerOfTwo
        provider_options.do_copy_in_default_stream = 0;
        // provider_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

        session_options.AppendExecutionProvider_CUDA(provider_options);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.SetLogSeverityLevel(log_level);
        session_options.SetLogId(name);

        session.reset(new Ort::Session(ort_env(), model_path, session_options));

        Ort::AllocatorWithDefaultOptions allocator;
        for (int i = 0; i < session->GetInputCount(); ++i)
        {
          inputs.push_back(Ort::Value(nullptr));
          input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
        }
        for (int i = 0; i < session->GetOutputCount(); ++i)
        {
          output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
        }

        std::transform(input_names.begin(), input_names.end(), std::back_inserter(input_names_cstr), [](std::string const &s)
                       { return s.c_str(); });

        std::transform(output_names.begin(), output_names.end(), std::back_inserter(output_names_cstr), [](std::string const &s)
                       { return s.c_str(); });
      }

      template <typename T>
      void set_input(std::string const &name, std::vector<T> &input, std::vector<int64_t> const &shape)
      {
        size_t idx = std::find(input_names.begin(), input_names.end(), name) - input_names.begin();
        inputs[idx] = Ort::Value::CreateTensor<T>(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPUInput), input.data(), input.size(), shape.data(), shape.size());
      }

      const std::vector<Ort::Value> infer()
      {
        return session->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), inputs.data(), input_names.size(), output_names_cstr.data(), output_names.size());
      }

      inline const std::vector<std::string> &get_input_names() const
      {
        return input_names;
      }

      inline const std::vector<std::string> &get_output_names() const
      {
        return output_names;
      }

      inline const size_t get_output_index(std::string const &name) const
      {
        return std::find(output_names.begin(), output_names.end(), name) - output_names.begin();
      }
    };
  }
}
#endif