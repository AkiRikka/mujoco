#include "experimental/nelif/onnx_backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifndef MUJOCO_NELIF_HAS_ONNXRUNTIME
#define MUJOCO_NELIF_HAS_ONNXRUNTIME 0
#endif

#if MUJOCO_NELIF_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace mujoco::nelif {
namespace {

constexpr int kRgbaChannels = 4;

using InputClock = std::chrono::steady_clock;
using InputTimePoint = InputClock::time_point;

double InputElapsedMs(InputTimePoint start, InputTimePoint end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void AddInputElapsed(double* total, InputTimePoint start) {
  if (total) {
    *total += InputElapsedMs(start, InputClock::now());
  }
}

float ClampFloat(float value, float lo, float hi) {
  return std::min(std::max(value, lo), hi);
}

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

#if MUJOCO_NELIF_HAS_ONNXRUNTIME
void ConfigureExecutionProvider(Ort::SessionOptions* session_options,
                                const OnnxIndirectConfig& config) {
  const std::string provider = LowerAscii(config.execution_provider);
  if (provider.empty() || provider == "cpu") {
    return;
  }
  if (provider == "coreml") {
    session_options->AppendExecutionProvider(
        "CoreML",
        {
            {"MLComputeUnits", "CPUAndGPU"},
            {"ModelFormat", "MLProgram"},
            {"RequireStaticInputShapes", "1"},
            {"EnableOnSubgraphs", "1"},
        });
    return;
  }
  if (provider == "cuda") {
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update({{"device_id", "0"}});
    session_options->AppendExecutionProvider_CUDA_V2(*cuda_options);
    return;
  }
  if (provider == "tensorrt" || provider == "trt") {
    Ort::TensorRTProviderOptions trt_options;
    trt_options.Update({
        {"device_id", "0"},
        {"trt_fp16_enable", "1"},
        {"trt_engine_cache_enable", "1"},
        {"trt_timing_cache_enable", "1"},
    });
    session_options->AppendExecutionProvider_TensorRT_V2(*trt_options);

    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update({{"device_id", "0"}});
    session_options->AppendExecutionProvider_CUDA_V2(*cuda_options);
    return;
  }
  throw std::invalid_argument("Unsupported ONNX execution provider: " +
                              config.execution_provider +
                              " (expected cpu, coreml, cuda, or tensorrt)");
}
#endif

float ReadChannelBilinearFlippedY(const std::vector<float>& rgba, int width, int height,
                                  float x, float y_top_origin, int channel) {
  if (rgba.empty() || width <= 0 || height <= 0) {
    return 0.0f;
  }
  const float x_clamped = ClampFloat(x, 0.0f, static_cast<float>(width - 1));
  const float y_clamped = ClampFloat(y_top_origin, 0.0f, static_cast<float>(height - 1));
  const int x0 = static_cast<int>(std::floor(x_clamped));
  const int y0_top = static_cast<int>(std::floor(y_clamped));
  const int x1 = std::min(x0 + 1, width - 1);
  const int y1_top = std::min(y0_top + 1, height - 1);
  const float tx = x_clamped - static_cast<float>(x0);
  const float ty = y_clamped - static_cast<float>(y0_top);

  auto sample = [&](int sx, int sy_top) {
    const int sy_gl = height - 1 - sy_top;
    const size_t index =
        (static_cast<size_t>(sy_gl) * static_cast<size_t>(width) + sx) * kRgbaChannels + channel;
    return rgba[index];
  };

  const float v00 = sample(x0, y0_top);
  const float v10 = sample(x1, y0_top);
  const float v01 = sample(x0, y1_top);
  const float v11 = sample(x1, y1_top);
  const float vx0 = v00 * (1.0f - tx) + v10 * tx;
  const float vx1 = v01 * (1.0f - tx) + v11 * tx;
  return vx0 * (1.0f - ty) + vx1 * ty;
}

void ResizeRgbTopOrigin(const std::vector<float>& rgba, int src_width, int src_height,
                        int dst_width, int dst_height, std::vector<float>* out) {
  out->assign(static_cast<size_t>(dst_width) * dst_height * 3, 0.0f);
  const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    for (int x = 0; x < dst_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const size_t dst = (static_cast<size_t>(y) * dst_width + x) * 3;
      out->at(dst + 0) = ReadChannelBilinearFlippedY(rgba, src_width, src_height, src_x, src_y, 0);
      out->at(dst + 1) = ReadChannelBilinearFlippedY(rgba, src_width, src_height, src_x, src_y, 1);
      out->at(dst + 2) = ReadChannelBilinearFlippedY(rgba, src_width, src_height, src_x, src_y, 2);
    }
  }
}

void ResizeRgbaTopOrigin(const std::vector<float>& rgba, int src_width, int src_height,
                         int dst_width, int dst_height, std::vector<float>* out) {
  out->assign(static_cast<size_t>(dst_width) * dst_height * 4, 0.0f);
  const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    for (int x = 0; x < dst_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const size_t dst = (static_cast<size_t>(y) * dst_width + x) * 4;
      for (int c = 0; c < 4; ++c) {
        out->at(dst + c) =
            ReadChannelBilinearFlippedY(rgba, src_width, src_height, src_x, src_y, c);
      }
    }
  }
}

void ResizeScalarTopOrigin(const std::vector<float>& rgba, int src_width, int src_height,
                           int dst_width, int dst_height, int channel,
                           std::vector<float>* out) {
  out->assign(static_cast<size_t>(dst_width) * dst_height, 0.0f);
  const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    for (int x = 0; x < dst_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      out->at(static_cast<size_t>(y) * dst_width + x) =
          ReadChannelBilinearFlippedY(rgba, src_width, src_height, src_x, src_y, channel);
    }
  }
}

float ReadChannelTopOriginNearest(const std::vector<float>& rgba, int width, int height,
                                  int x, int y_top, int channel) {
  const int y_gl = height - 1 - y_top;
  const size_t index =
      (static_cast<size_t>(y_gl) * static_cast<size_t>(width) + x) * kRgbaChannels + channel;
  return rgba[index];
}

void DownsampleLightRgbaTopOrigin(const std::vector<float>& rgba, int src_width,
                                  int src_height, int target_face_size,
                                  std::vector<float>* out) {
  if (src_width <= 0 || src_height != src_width * 6 || target_face_size <= 0) {
    ResizeRgbaTopOrigin(rgba, src_width, src_height, target_face_size,
                        target_face_size * 6, out);
    return;
  }
  if (src_width == target_face_size) {
    ResizeRgbaTopOrigin(rgba, src_width, src_height, target_face_size,
                        target_face_size * 6, out);
    return;
  }
  if (src_width % target_face_size != 0) {
    ResizeRgbaTopOrigin(rgba, src_width, src_height, target_face_size,
                        target_face_size * 6, out);
    return;
  }

  const int factor = src_width / target_face_size;
  const int dst_width = target_face_size;
  const int dst_height = target_face_size * 6;
  out->assign(static_cast<size_t>(dst_width) * dst_height * 4, 0.0f);
  const float inv_area = 1.0f / static_cast<float>(factor * factor);
  for (int face = 0; face < 6; ++face) {
    for (int y = 0; y < target_face_size; ++y) {
      for (int x = 0; x < target_face_size; ++x) {
        float sum[4] = {};
        for (int yy = 0; yy < factor; ++yy) {
          const int src_y = face * src_width + y * factor + yy;
          for (int xx = 0; xx < factor; ++xx) {
            const int src_x = x * factor + xx;
            for (int c = 0; c < 4; ++c) {
              sum[c] += ReadChannelTopOriginNearest(rgba, src_width, src_height,
                                                    src_x, src_y, c);
            }
          }
        }
        const size_t dst =
            (static_cast<size_t>(face * target_face_size + y) * dst_width + x) * 4;
        for (int c = 0; c < 4; ++c) {
          out->at(dst + c) = sum[c] * inv_area;
        }
      }
    }
  }
}

void MaskedRgbFromRgba(const std::vector<float>& rgba, const std::vector<float>& mask_rgba,
                       int width, int height, std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height * 3, 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y) * width + x) * 4;
      const size_t dst = (static_cast<size_t>(y) * width + x) * 3;
      if (mask_rgba[src + 3] > 0.5f) {
        out->at(dst + 0) = rgba[src + 0];
        out->at(dst + 1) = rgba[src + 1];
        out->at(dst + 2) = rgba[src + 2];
      }
    }
  }
}

void RoughnessFromGlossTopOrigin(const std::vector<float>& gloss_rgba, int src_width,
                                 int src_height, const std::vector<float>& position_rgba,
                                 int dst_width, int dst_height,
                                 std::vector<float>* out) {
  out->assign(static_cast<size_t>(dst_width) * dst_height, 0.0f);
  const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    const int y0_top = static_cast<int>(std::floor(ClampFloat(src_y, 0.0f,
                                                            static_cast<float>(src_height - 1))));
    const int y1_top = std::min(y0_top + 1, src_height - 1);
    const float ty = ClampFloat(src_y, 0.0f, static_cast<float>(src_height - 1)) -
                     static_cast<float>(y0_top);
    for (int x = 0; x < dst_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const int x0 = static_cast<int>(std::floor(ClampFloat(src_x, 0.0f,
                                                           static_cast<float>(src_width - 1))));
      const int x1 = std::min(x0 + 1, src_width - 1);
      const float tx = ClampFloat(src_x, 0.0f, static_cast<float>(src_width - 1)) -
                       static_cast<float>(x0);

      auto sample = [&](int sx, int sy_top) {
        const float alpha = ReadChannelTopOriginNearest(position_rgba, src_width, src_height,
                                                        sx, sy_top, 3);
        if (alpha <= 0.5f) {
          return 0.0f;
        }
        const float gloss = ReadChannelTopOriginNearest(gloss_rgba, src_width, src_height,
                                                        sx, sy_top, 0);
        return ClampFloat(1.0f - gloss, 0.0f, 1.0f);
      };

      const float v00 = sample(x0, y0_top);
      const float v10 = sample(x1, y0_top);
      const float v01 = sample(x0, y1_top);
      const float v11 = sample(x1, y1_top);
      const float vx0 = v00 * (1.0f - tx) + v10 * tx;
      const float vx1 = v01 * (1.0f - tx) + v11 * tx;
      out->at(static_cast<size_t>(y) * dst_width + x) = vx0 * (1.0f - ty) + vx1 * ty;
    }
  }
}

void ResizeDepthFromPositionTopOrigin(const std::vector<float>& position_rgba, int src_width,
                                      int src_height, int dst_width, int dst_height,
                                      const float camera_pos[3], std::vector<float>* out) {
  out->assign(static_cast<size_t>(dst_width) * dst_height, 0.0f);
  const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    const int y0_top = static_cast<int>(std::floor(ClampFloat(src_y, 0.0f,
                                                            static_cast<float>(src_height - 1))));
    const int y1_top = std::min(y0_top + 1, src_height - 1);
    const float ty = ClampFloat(src_y, 0.0f, static_cast<float>(src_height - 1)) -
                     static_cast<float>(y0_top);
    for (int x = 0; x < dst_width; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const int x0 = static_cast<int>(std::floor(ClampFloat(src_x, 0.0f,
                                                           static_cast<float>(src_width - 1))));
      const int x1 = std::min(x0 + 1, src_width - 1);
      const float tx = ClampFloat(src_x, 0.0f, static_cast<float>(src_width - 1)) -
                       static_cast<float>(x0);

      auto sample = [&](int sx, int sy_top) {
        const float px = ReadChannelTopOriginNearest(position_rgba, src_width, src_height,
                                                     sx, sy_top, 0);
        const float py = ReadChannelTopOriginNearest(position_rgba, src_width, src_height,
                                                     sx, sy_top, 1);
        const float pz = ReadChannelTopOriginNearest(position_rgba, src_width, src_height,
                                                     sx, sy_top, 2);
        const float dx = px - camera_pos[0];
        const float dy = py - camera_pos[1];
        const float dz = pz - camera_pos[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
      };

      const float v00 = sample(x0, y0_top);
      const float v10 = sample(x1, y0_top);
      const float v01 = sample(x0, y1_top);
      const float v11 = sample(x1, y1_top);
      const float vx0 = v00 * (1.0f - tx) + v10 * tx;
      const float vx1 = v01 * (1.0f - tx) + v11 * tx;
      out->at(static_cast<size_t>(y) * dst_width + x) = vx0 * (1.0f - ty) + vx1 * ty;
    }
  }
}

void RgbTopOriginToGlRgba(const float* rgb, int width, int height,
                          std::vector<float>* rgba) {
  rgba->assign(static_cast<size_t>(width) * height * 4, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_top) * width + x) * 3;
      const size_t dst = (static_cast<size_t>(y_gl) * width + x) * 4;
      rgba->at(dst + 0) = rgb[src + 0];
      rgba->at(dst + 1) = rgb[src + 1];
      rgba->at(dst + 2) = rgb[src + 2];
      rgba->at(dst + 3) = 1.0f;
    }
  }
}

void TensorTopOriginToGlRgba(const float* data, int width, int height, int channels,
                             bool clamp_unit, std::vector<float>* rgba) {
  rgba->assign(static_cast<size_t>(width) * height * 4, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_top) * width + x) * channels;
      const size_t dst = (static_cast<size_t>(y_gl) * width + x) * 4;
      const float r = channels == 1 ? data[src] : data[src + 0];
      const float g = channels == 1 ? data[src] : data[src + 1];
      const float b = channels == 1 ? data[src] : data[src + 2];
      rgba->at(dst + 0) = clamp_unit ? ClampFloat(r, 0.0f, 1.0f) : r;
      rgba->at(dst + 1) = clamp_unit ? ClampFloat(g, 0.0f, 1.0f) : g;
      rgba->at(dst + 2) = clamp_unit ? ClampFloat(b, 0.0f, 1.0f) : b;
      rgba->at(dst + 3) = 1.0f;
    }
  }
}

const std::array<const char*, kOnnxRuntimeInputCount>& RuntimeInputNames() {
  static const std::array<const char*, kOnnxRuntimeInputCount> input_names = {
      "position",
      "normal",
      "view_dir",
      "albedo",
      "specular",
      "roughness",
      "light_dir",
      "pixel_emitter_distance",
      "occluder_emitter_distance",
      "depth",
      "light_position",
      "light_normal",
      "light_albedo",
      "max_scale",
  };
  return input_names;
}

void FillRuntimeInputShapes(int screen_w, int screen_h, int rsm_w, int rsm_h,
                            OnnxRuntimeInputs* out) {
  out->shapes = {
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 1},
      std::vector<int64_t>{1, screen_h, screen_w, 3},
      std::vector<int64_t>{1, screen_h, screen_w, 1},
      std::vector<int64_t>{1, screen_h, screen_w, 1},
      std::vector<int64_t>{1, screen_h, screen_w, 1},
      std::vector<int64_t>{1, rsm_h, rsm_w, 3},
      std::vector<int64_t>{1, rsm_h, rsm_w, 3},
      std::vector<int64_t>{1, rsm_h, rsm_w, 3},
      std::vector<int64_t>{1, 3},
  };
}

bool BuildRuntimeInputBuffers(const GBufferPass& gbuffer,
                              const ScreenSpaceLightPass& screen_space_light,
                              const float camera_pos[3],
                              const OnnxIndirectConfig& config,
                              OnnxRuntimeInputs* out,
                              std::string* error,
                              OnnxRuntimeInputTiming* timing) {
  if (timing) {
    *timing = OnnxRuntimeInputTiming();
  }
  const int screen_w = config.screen_width;
  const int screen_h = config.screen_height;
  const int rsm_face = config.rsm_face_size;
  const int rsm_w = rsm_face;
  const int rsm_h = rsm_face * 6;

  if (screen_w <= 0 || screen_h <= 0 || rsm_face <= 0) {
    if (error) {
      *error = "Invalid ONNX runtime input dimensions.";
    }
    return false;
  }

  std::vector<float> rgba;
  std::vector<float> position_rgba;

  auto read_gbuffer = [&](GBufferAttachment attachment, int index, bool scalar,
                          int channel = 0) -> bool {
    InputTimePoint stage_start = InputClock::now();
    if (!gbuffer.Readback(attachment, &rgba)) {
      return false;
    }
    AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
    stage_start = InputClock::now();
    if (scalar) {
      ResizeScalarTopOrigin(rgba, gbuffer.width(), gbuffer.height(), screen_w, screen_h,
                            channel, &out->buffers[index]);
    } else {
      ResizeRgbTopOrigin(rgba, gbuffer.width(), gbuffer.height(), screen_w, screen_h,
                         &out->buffers[index]);
    }
    AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);
    return true;
  };

  auto read_screen = [&](ScreenSpaceLightAttachment attachment, int index, bool scalar,
                         int channel = 0) -> bool {
    InputTimePoint stage_start = InputClock::now();
    if (!screen_space_light.Readback(attachment, &rgba)) {
      return false;
    }
    AddInputElapsed(timing ? &timing->screen_read_ms : nullptr, stage_start);
    stage_start = InputClock::now();
    if (scalar) {
      ResizeScalarTopOrigin(rgba, screen_space_light.width(), screen_space_light.height(),
                            screen_w, screen_h, channel, &out->buffers[index]);
    } else {
      ResizeRgbTopOrigin(rgba, screen_space_light.width(), screen_space_light.height(),
                         screen_w, screen_h, &out->buffers[index]);
    }
    AddInputElapsed(timing ? &timing->screen_resize_ms : nullptr, stage_start);
    return true;
  };

  InputTimePoint stage_start = InputClock::now();
  if (!gbuffer.Readback(GBufferAttachment::kPosition, &position_rgba)) {
    if (error) {
      *error = "Failed to read Position attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  ResizeRgbTopOrigin(position_rgba, gbuffer.width(), gbuffer.height(), screen_w, screen_h,
                     &out->buffers[0]);
  AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);

  if (!read_gbuffer(GBufferAttachment::kNormal, 1, false) ||
      !read_gbuffer(GBufferAttachment::kOutDir, 2, false) ||
      !read_gbuffer(GBufferAttachment::kDiffuse, 3, false) ||
      !read_gbuffer(GBufferAttachment::kReflect, 4, false)) {
    if (error) {
      *error = "Failed to read G-buffer RGB attachment.";
    }
    return false;
  }

  stage_start = InputClock::now();
  if (!gbuffer.Readback(GBufferAttachment::kGloss, &rgba)) {
    if (error) {
      *error = "Failed to read Gloss attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  RoughnessFromGlossTopOrigin(rgba, gbuffer.width(), gbuffer.height(), position_rgba,
                              screen_w, screen_h, &out->buffers[5]);
  AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);

  if (!read_screen(ScreenSpaceLightAttachment::kSSLightVec, 6, false)) {
    if (error) {
      *error = "Failed to read screen-space light attachment.";
    }
    return false;
  }

  stage_start = InputClock::now();
  if (!screen_space_light.Readback(ScreenSpaceLightAttachment::kSSLightDepth, &rgba)) {
    if (error) {
      *error = "Failed to read screen-space light depth attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->screen_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  ResizeScalarTopOrigin(rgba, screen_space_light.width(), screen_space_light.height(),
                        screen_w, screen_h, 1, &out->buffers[7]);
  ResizeScalarTopOrigin(rgba, screen_space_light.width(), screen_space_light.height(),
                        screen_w, screen_h, 0, &out->buffers[8]);
  AddInputElapsed(timing ? &timing->screen_resize_ms : nullptr, stage_start);

  stage_start = InputClock::now();
  ResizeDepthFromPositionTopOrigin(position_rgba, gbuffer.width(), gbuffer.height(),
                                   screen_w, screen_h, camera_pos, &out->buffers[9]);
  AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);

  std::vector<float> light_position_rgba;
  std::vector<float> light_normal_rgba;
  std::vector<float> light_albedo_rgba;
  stage_start = InputClock::now();
  if (!screen_space_light.Readback(LightSpaceAttachment::kSMPosition, &rgba)) {
    if (error) {
      *error = "Failed to read smPosition attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->light_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  DownsampleLightRgbaTopOrigin(rgba, screen_space_light.light_space_width(),
                               screen_space_light.light_space_height(), rsm_face,
                               &light_position_rgba);
  AddInputElapsed(timing ? &timing->light_downsample_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  if (!screen_space_light.Readback(LightSpaceAttachment::kSMNormal, &rgba)) {
    if (error) {
      *error = "Failed to read smNormal attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->light_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  DownsampleLightRgbaTopOrigin(rgba, screen_space_light.light_space_width(),
                               screen_space_light.light_space_height(), rsm_face,
                               &light_normal_rgba);
  AddInputElapsed(timing ? &timing->light_downsample_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  if (!screen_space_light.Readback(LightSpaceAttachment::kSMFlux, &rgba)) {
    if (error) {
      *error = "Failed to read smFlux attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->light_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  DownsampleLightRgbaTopOrigin(rgba, screen_space_light.light_space_width(),
                               screen_space_light.light_space_height(), rsm_face,
                               &light_albedo_rgba);
  AddInputElapsed(timing ? &timing->light_downsample_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  MaskedRgbFromRgba(light_position_rgba, light_normal_rgba, rsm_w, rsm_h,
                    &out->buffers[10]);
  MaskedRgbFromRgba(light_normal_rgba, light_normal_rgba, rsm_w, rsm_h,
                    &out->buffers[11]);
  MaskedRgbFromRgba(light_albedo_rgba, light_normal_rgba, rsm_w, rsm_h,
                    &out->buffers[12]);
  AddInputElapsed(timing ? &timing->light_mask_ms : nullptr, stage_start);

  stage_start = InputClock::now();
  out->buffers[13] = {config.max_scale[0], config.max_scale[1], config.max_scale[2]};
  FillRuntimeInputShapes(screen_w, screen_h, rsm_w, rsm_h, out);
  AddInputElapsed(timing ? &timing->finalize_ms : nullptr, stage_start);
  return true;
}

}  // namespace

bool BuildOnnxRuntimeInputs(const GBufferPass& gbuffer,
                            const ScreenSpaceLightPass& screen_space_light,
                            const float camera_pos[3],
                            const OnnxIndirectConfig& config,
                            OnnxRuntimeInputs* out,
                            std::string* error,
                            OnnxRuntimeInputTiming* timing) {
  return BuildRuntimeInputBuffers(gbuffer, screen_space_light, camera_pos, config, out, error,
                                  timing);
}

struct OnnxIndirectBackend::Impl {
  OnnxIndirectConfig config;
  int last_output_width = 0;
  int last_output_height = 0;
  std::string status = "ONNX Runtime backend is not compiled in";

#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "nelif_onnx"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
};

OnnxIndirectBackend::OnnxIndirectBackend() : impl_(new Impl) {}

OnnxIndirectBackend::~OnnxIndirectBackend() {
  Shutdown();
  delete impl_;
  impl_ = nullptr;
}

bool OnnxIndirectBackend::Init(const OnnxIndirectConfig& config, std::string* error) {
  impl_->config = config;
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  try {
    impl_->session_options.SetIntraOpNumThreads(1);
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    ConfigureExecutionProvider(&impl_->session_options, config);
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, config.model_path.c_str(), impl_->session_options);
    impl_->status = "ONNX Runtime indirect backend enabled provider=" +
                    LowerAscii(config.execution_provider);
    return true;
  } catch (const Ort::Exception& ex) {
    if (error) {
      *error = ex.what();
    }
    impl_->status = "Failed to initialize ONNX Runtime indirect backend";
    return false;
  } catch (const std::exception& ex) {
    if (error) {
      *error = ex.what();
    }
    impl_->status = "Failed to initialize ONNX Runtime indirect backend";
    return false;
  }
#else
  if (error) {
    *error =
        "ONNX Runtime backend was not compiled. Configure with "
        "-DMUJOCO_NELIF_ENABLE_ONNXRUNTIME=ON and "
        "-DMUJOCO_NELIF_ONNXRUNTIME_ROOT=/path/to/onnxruntime.";
  }
  impl_->status = "ONNX Runtime backend is not compiled in";
  return false;
#endif
}

bool OnnxIndirectBackend::IsEnabled() const {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  return impl_ && impl_->session != nullptr;
#else
  return false;
#endif
}

const char* OnnxIndirectBackend::Status() const {
  return impl_ ? impl_->status.c_str() : "ONNX Runtime backend is shut down";
}

void OnnxIndirectBackend::Shutdown() {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  if (impl_) {
    impl_->session.reset();
    impl_->status = "ONNX Runtime backend is shut down";
  }
#endif
}

bool OnnxIndirectBackend::Run(const GBufferPass& gbuffer,
                              const ScreenSpaceLightPass& screen_space_light,
                              const float camera_pos[3],
                              std::vector<float>* indirect_rgba,
                              std::string* error) {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  OnnxRuntimeInputs input_buffers;
  if (!BuildOnnxRuntimeInputs(gbuffer, screen_space_light, camera_pos, impl_->config,
                              &input_buffers, error)) {
    return false;
  }
  return Run(input_buffers, indirect_rgba, error);
#else
  if (error) {
    *error = "ONNX Runtime backend was not compiled.";
  }
  (void)gbuffer;
  (void)screen_space_light;
  (void)camera_pos;
  (void)indirect_rgba;
  return false;
#endif
}

bool OnnxIndirectBackend::Run(OnnxRuntimeInputs& inputs,
                              std::vector<float>* indirect_rgba,
                              std::string* error) {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  if (!IsEnabled()) {
    if (error) {
      *error = "ONNX Runtime backend is not initialized.";
    }
    return false;
  }

  auto& buffers = inputs.buffers;
  const auto& shapes = inputs.shapes;
  const auto& input_names = RuntimeInputNames();

  std::array<Ort::Value, kOnnxRuntimeInputCount> tensors = {
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[0].data(),
                                      buffers[0].size(), shapes[0].data(), shapes[0].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[1].data(),
                                      buffers[1].size(), shapes[1].data(), shapes[1].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[2].data(),
                                      buffers[2].size(), shapes[2].data(), shapes[2].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[3].data(),
                                      buffers[3].size(), shapes[3].data(), shapes[3].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[4].data(),
                                      buffers[4].size(), shapes[4].data(), shapes[4].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[5].data(),
                                      buffers[5].size(), shapes[5].data(), shapes[5].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[6].data(),
                                      buffers[6].size(), shapes[6].data(), shapes[6].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[7].data(),
                                      buffers[7].size(), shapes[7].data(), shapes[7].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[8].data(),
                                      buffers[8].size(), shapes[8].data(), shapes[8].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[9].data(),
                                      buffers[9].size(), shapes[9].data(), shapes[9].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[10].data(),
                                      buffers[10].size(), shapes[10].data(), shapes[10].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[11].data(),
                                      buffers[11].size(), shapes[11].data(), shapes[11].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[12].data(),
                                      buffers[12].size(), shapes[12].data(), shapes[12].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[13].data(),
                                      buffers[13].size(), shapes[13].data(), shapes[13].size()),
  };

  const std::array<const char*, 1> output_names = {"indirect_shading"};
  try {
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                      tensors.data(), input_names.size(),
                                      output_names.data(), output_names.size());
    if (outputs.empty()) {
      if (error) {
        *error = "ONNX Runtime returned no outputs.";
      }
      return false;
    }
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = info.GetShape();
    if (out_shape.size() != 4 || out_shape[0] != 1 || out_shape[3] != 3) {
      if (error) {
        std::ostringstream out;
        out << "Unexpected indirect output shape:";
        for (int64_t dim : out_shape) {
          out << " " << dim;
        }
        *error = out.str();
      }
      return false;
    }
    const int out_h = static_cast<int>(out_shape[1]);
    const int out_w = static_cast<int>(out_shape[2]);
    RgbTopOriginToGlRgba(outputs[0].GetTensorData<float>(), out_w, out_h, indirect_rgba);
    impl_->last_output_width = out_w;
    impl_->last_output_height = out_h;
    return true;
  } catch (const Ort::Exception& ex) {
    if (error) {
      *error = ex.what();
    }
    return false;
  }
#else
  if (error) {
    *error = "ONNX Runtime backend was not compiled.";
  }
  (void)inputs;
  (void)indirect_rgba;
  return false;
#endif
}

int OnnxIndirectBackend::output_width() const {
  if (!impl_) {
    return 0;
  }
  return impl_->last_output_width > 0 ? impl_->last_output_width : impl_->config.screen_width;
}

int OnnxIndirectBackend::output_height() const {
  if (!impl_) {
    return 0;
  }
  return impl_->last_output_height > 0 ? impl_->last_output_height : impl_->config.screen_height;
}

struct OnnxShadowBackend::Impl {
  OnnxShadowConfig config;
  int last_output_width = 0;
  int last_output_height = 0;
  std::string status = "ONNX Runtime backend is not compiled in";

#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "nelif_shadow_onnx"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
};

OnnxShadowBackend::OnnxShadowBackend() : impl_(new Impl) {}

OnnxShadowBackend::~OnnxShadowBackend() {
  Shutdown();
  delete impl_;
  impl_ = nullptr;
}

bool OnnxShadowBackend::Init(const OnnxShadowConfig& config, std::string* error) {
  impl_->config = config;
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  try {
    impl_->session_options.SetIntraOpNumThreads(1);
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    ConfigureExecutionProvider(&impl_->session_options, config);
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, config.model_path.c_str(), impl_->session_options);
    impl_->status = "ONNX Runtime shadow backend enabled provider=" +
                    LowerAscii(config.execution_provider);
    return true;
  } catch (const Ort::Exception& ex) {
    if (error) {
      *error = ex.what();
    }
    impl_->status = "Failed to initialize ONNX Runtime shadow backend";
    return false;
  } catch (const std::exception& ex) {
    if (error) {
      *error = ex.what();
    }
    impl_->status = "Failed to initialize ONNX Runtime shadow backend";
    return false;
  }
#else
  if (error) {
    *error =
        "ONNX Runtime backend was not compiled. Configure with "
        "-DMUJOCO_NELIF_ENABLE_ONNXRUNTIME=ON and "
        "-DMUJOCO_NELIF_ONNXRUNTIME_ROOT=/path/to/onnxruntime.";
  }
  impl_->status = "ONNX Runtime backend is not compiled in";
  return false;
#endif
}

bool OnnxShadowBackend::IsEnabled() const {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  return impl_ && impl_->session != nullptr;
#else
  return false;
#endif
}

const char* OnnxShadowBackend::Status() const {
  return impl_ ? impl_->status.c_str() : "ONNX Runtime backend is shut down";
}

void OnnxShadowBackend::Shutdown() {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  if (impl_) {
    impl_->session.reset();
    impl_->status = "ONNX Runtime backend is shut down";
  }
#endif
}

bool OnnxShadowBackend::Run(const GBufferPass& gbuffer,
                            const ScreenSpaceLightPass& screen_space_light,
                            const float camera_pos[3],
                            std::vector<float>* shadow_rgba,
                            std::string* error) {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  OnnxRuntimeInputs input_buffers;
  if (!BuildOnnxRuntimeInputs(gbuffer, screen_space_light, camera_pos, impl_->config,
                              &input_buffers, error)) {
    return false;
  }
  return Run(input_buffers, shadow_rgba, error);
#else
  if (error) {
    *error = "ONNX Runtime backend was not compiled.";
  }
  (void)gbuffer;
  (void)screen_space_light;
  (void)camera_pos;
  (void)shadow_rgba;
  return false;
#endif
}

bool OnnxShadowBackend::Run(OnnxRuntimeInputs& inputs,
                            std::vector<float>* shadow_rgba,
                            std::string* error) {
#if MUJOCO_NELIF_HAS_ONNXRUNTIME
  if (!IsEnabled()) {
    if (error) {
      *error = "ONNX Runtime backend is not initialized.";
    }
    return false;
  }

  auto& buffers = inputs.buffers;
  const auto& shapes = inputs.shapes;
  const auto& input_names = RuntimeInputNames();

  std::array<Ort::Value, kOnnxRuntimeInputCount> tensors = {
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[0].data(),
                                      buffers[0].size(), shapes[0].data(), shapes[0].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[1].data(),
                                      buffers[1].size(), shapes[1].data(), shapes[1].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[2].data(),
                                      buffers[2].size(), shapes[2].data(), shapes[2].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[3].data(),
                                      buffers[3].size(), shapes[3].data(), shapes[3].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[4].data(),
                                      buffers[4].size(), shapes[4].data(), shapes[4].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[5].data(),
                                      buffers[5].size(), shapes[5].data(), shapes[5].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[6].data(),
                                      buffers[6].size(), shapes[6].data(), shapes[6].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[7].data(),
                                      buffers[7].size(), shapes[7].data(), shapes[7].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[8].data(),
                                      buffers[8].size(), shapes[8].data(), shapes[8].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[9].data(),
                                      buffers[9].size(), shapes[9].data(), shapes[9].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[10].data(),
                                      buffers[10].size(), shapes[10].data(), shapes[10].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[11].data(),
                                      buffers[11].size(), shapes[11].data(), shapes[11].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[12].data(),
                                      buffers[12].size(), shapes[12].data(), shapes[12].size()),
      Ort::Value::CreateTensor<float>(impl_->memory_info, buffers[13].data(),
                                      buffers[13].size(), shapes[13].data(), shapes[13].size()),
  };

  const std::array<const char*, 1> output_names = {"shadow"};
  try {
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                      tensors.data(), input_names.size(),
                                      output_names.data(), output_names.size());
    if (outputs.empty()) {
      if (error) {
        *error = "ONNX Runtime returned no outputs.";
      }
      return false;
    }
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = info.GetShape();
    if (out_shape.size() != 4 || out_shape[0] != 1 ||
        (out_shape[3] != 1 && out_shape[3] != 3)) {
      if (error) {
        std::ostringstream out;
        out << "Unexpected shadow output shape:";
        for (int64_t dim : out_shape) {
          out << " " << dim;
        }
        *error = out.str();
      }
      return false;
    }
    const int out_h = static_cast<int>(out_shape[1]);
    const int out_w = static_cast<int>(out_shape[2]);
    const int out_c = static_cast<int>(out_shape[3]);
    TensorTopOriginToGlRgba(outputs[0].GetTensorData<float>(), out_w, out_h, out_c,
                            /*clamp_unit=*/true, shadow_rgba);
    impl_->last_output_width = out_w;
    impl_->last_output_height = out_h;
    return true;
  } catch (const Ort::Exception& ex) {
    if (error) {
      *error = ex.what();
    }
    return false;
  }
#else
  if (error) {
    *error = "ONNX Runtime backend was not compiled.";
  }
  (void)inputs;
  (void)shadow_rgba;
  return false;
#endif
}

int OnnxShadowBackend::output_width() const {
  if (!impl_) {
    return 0;
  }
  return impl_->last_output_width > 0 ? impl_->last_output_width : impl_->config.screen_width;
}

int OnnxShadowBackend::output_height() const {
  if (!impl_) {
    return 0;
  }
  return impl_->last_output_height > 0 ? impl_->last_output_height : impl_->config.screen_height;
}

}  // namespace mujoco::nelif
