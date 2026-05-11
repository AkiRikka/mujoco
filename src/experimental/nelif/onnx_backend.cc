#include "experimental/nelif/onnx_backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifndef MUJOCO_NELIF_HAS_ONNXRUNTIME
#define MUJOCO_NELIF_HAS_ONNXRUNTIME 0
#endif

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

#ifndef GL_READ_ONLY
#define GL_READ_ONLY 0x88B8
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

class RgbaReadbackStager {
 public:
  bool Read(GLuint source_texture, int source_width, int source_height,
            int target_width, int target_height, GLenum filter,
            bool use_pbo, std::vector<float>* rgba) {
    if (!source_texture || source_width <= 0 || source_height <= 0 ||
        target_width <= 0 || target_height <= 0 || !rgba) {
      return false;
    }
    Ensure(target_width, target_height);

    rgba->assign(static_cast<size_t>(target_width) * target_height * 4, 0.0f);

    GLint previous_read_fbo = 0;
    GLint previous_draw_fbo = 0;
    GLint previous_texture = 0;
    GLint previous_pack_buffer = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previous_read_fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous_draw_fbo);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &previous_pack_buffer);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, source_texture, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo_);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, staging_texture_, 0);
    GLenum draw_buffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &draw_buffer);

    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
        glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      Restore(previous_read_fbo, previous_draw_fbo, previous_texture,
              previous_pack_buffer);
      return false;
    }

    glBlitFramebuffer(0, 0, source_width, source_height,
                      0, 0, target_width, target_height,
                      GL_COLOR_BUFFER_BIT, filter);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, draw_fbo_);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    bool ok = false;
    if (use_pbo) {
      ok = ReadPixelsPbo(source_texture, target_width, target_height, rgba);
    } else {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
      glReadPixels(0, 0, target_width, target_height, GL_RGBA, GL_FLOAT, rgba->data());
      ok = glGetError() == GL_NO_ERROR;
    }

    Restore(previous_read_fbo, previous_draw_fbo, previous_texture,
            previous_pack_buffer);
    return ok;
  }

 private:
  struct PboSlot {
    GLuint pbo = 0;
    size_t bytes = 0;
    bool pending = false;
  };

  void Ensure(int width, int height) {
    if (!read_fbo_) {
      glGenFramebuffers(1, &read_fbo_);
    }
    if (!draw_fbo_) {
      glGenFramebuffers(1, &draw_fbo_);
    }
    if (staging_texture_ && width == width_ && height == height_) {
      return;
    }
    if (!staging_texture_) {
      glGenTextures(1, &staging_texture_);
    }
    width_ = width;
    height_ = height;
    glBindTexture(GL_TEXTURE_2D, staging_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
  }

  bool ReadPixelsPbo(GLuint source_texture, int width, int height,
                     std::vector<float>* rgba) {
    const size_t bytes = static_cast<size_t>(width) * height * 4 * sizeof(float);
    PboSlot& slot = pbo_slots_[PboKey(source_texture, width, height)];
    if (!slot.pbo) {
      glGenBuffers(1, &slot.pbo);
    }
    if (slot.bytes != bytes) {
      slot.bytes = bytes;
      slot.pending = false;
      glBindBuffer(GL_PIXEL_PACK_BUFFER, slot.pbo);
      glBufferData(GL_PIXEL_PACK_BUFFER, slot.bytes, nullptr, GL_STREAM_READ);
    }

    if (slot.pending) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, slot.pbo);
      void* mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
      if (mapped) {
        std::memcpy(rgba->data(), mapped, slot.bytes);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
      }
      slot.pending = false;
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, slot.pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, slot.bytes, nullptr, GL_STREAM_READ);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, nullptr);
    const GLenum read_error = glGetError();
    slot.pending = read_error == GL_NO_ERROR;

    // First frame, or a missed previous map, keeps the zero-initialized output.
    // Warmup frames in benchmarks hide this startup latency.
    return read_error == GL_NO_ERROR;
  }

  static uint64_t PboKey(GLuint texture, int width, int height) {
    const uint64_t t = static_cast<uint64_t>(texture);
    const uint64_t w = static_cast<uint64_t>(static_cast<uint32_t>(width));
    const uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(height));
    return (t << 32) ^ (w << 16) ^ h;
  }

  void Restore(GLint read_fbo, GLint draw_fbo, GLint texture, GLint pack_buffer) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(read_fbo));
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(draw_fbo));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
    glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(pack_buffer));
  }

  GLuint read_fbo_ = 0;
  GLuint draw_fbo_ = 0;
  GLuint staging_texture_ = 0;
  int width_ = 0;
  int height_ = 0;
  std::unordered_map<uint64_t, PboSlot> pbo_slots_;
};

bool ReadTextureResizedRGBA32F(GLuint texture, int source_width, int source_height,
                               int target_width, int target_height, GLenum filter,
                               bool use_pbo, std::vector<float>* rgba) {
  static thread_local RgbaReadbackStager stager;
  return stager.Read(texture, source_width, source_height,
                     target_width, target_height, filter, use_pbo, rgba);
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

void CopyRgbTopOrigin(const std::vector<float>& rgba, int width, int height,
                      std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height * 3, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_gl) * width + x) * 4;
      const size_t dst = (static_cast<size_t>(y_top) * width + x) * 3;
      out->at(dst + 0) = rgba[src + 0];
      out->at(dst + 1) = rgba[src + 1];
      out->at(dst + 2) = rgba[src + 2];
    }
  }
}

void CopyRgbaTopOrigin(const std::vector<float>& rgba, int width, int height,
                       std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height * 4, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    const size_t src = static_cast<size_t>(y_gl) * width * 4;
    const size_t dst = static_cast<size_t>(y_top) * width * 4;
    std::copy_n(rgba.data() + src, static_cast<size_t>(width) * 4,
                out->data() + dst);
  }
}

void CopyScalarTopOrigin(const std::vector<float>& rgba, int width, int height,
                         int channel, std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_gl) * width + x) * 4 + channel;
      out->at(static_cast<size_t>(y_top) * width + x) = rgba[src];
    }
  }
}

void RoughnessFromGlossSameSizeTopOrigin(const std::vector<float>& gloss_rgba,
                                         int width, int height,
                                         const std::vector<float>& position_rgba,
                                         std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    const int y_gl = height - 1 - y_top;
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_gl) * width + x) * 4;
      const size_t mask_src = (static_cast<size_t>(y_top) * width + x) * 4;
      if (position_rgba[mask_src + 3] > 0.5f) {
        out->at(static_cast<size_t>(y_top) * width + x) =
            ClampFloat(1.0f - gloss_rgba[src], 0.0f, 1.0f);
      }
    }
  }
}

void DepthFromPositionSameSizeTopOrigin(const std::vector<float>& position_rgba,
                                        int width, int height,
                                        const float camera_pos[3],
                                        std::vector<float>* out) {
  out->assign(static_cast<size_t>(width) * height, 0.0f);
  for (int y_top = 0; y_top < height; ++y_top) {
    for (int x = 0; x < width; ++x) {
      const size_t src = (static_cast<size_t>(y_top) * width + x) * 4;
      const float dx = position_rgba[src + 0] - camera_pos[0];
      const float dy = position_rgba[src + 1] - camera_pos[1];
      const float dz = position_rgba[src + 2] - camera_pos[2];
      out->at(static_cast<size_t>(y_top) * width + x) =
          std::sqrt(dx * dx + dy * dy + dz * dz);
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

bool BuildRuntimeInputBuffersGpuStaging(const GBufferPass& gbuffer,
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
  std::vector<float> position_top_rgba;

  auto read_gbuffer = [&](GBufferAttachment attachment, int index, bool scalar,
                          int channel = 0) -> bool {
    InputTimePoint stage_start = InputClock::now();
    if (!ReadTextureResizedRGBA32F(gbuffer.Texture(attachment), gbuffer.width(),
                                   gbuffer.height(), screen_w, screen_h,
                                   GL_LINEAR, config.use_pbo_readback, &rgba)) {
      return false;
    }
    AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
    stage_start = InputClock::now();
    if (scalar) {
      CopyScalarTopOrigin(rgba, screen_w, screen_h, channel, &out->buffers[index]);
    } else {
      CopyRgbTopOrigin(rgba, screen_w, screen_h, &out->buffers[index]);
    }
    AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);
    return true;
  };

  auto read_screen = [&](ScreenSpaceLightAttachment attachment, int index, bool scalar,
                         int channel = 0) -> bool {
    InputTimePoint stage_start = InputClock::now();
    if (!ReadTextureResizedRGBA32F(screen_space_light.Texture(attachment),
                                   screen_space_light.width(),
                                   screen_space_light.height(), screen_w, screen_h,
                                   GL_LINEAR, config.use_pbo_readback, &rgba)) {
      return false;
    }
    AddInputElapsed(timing ? &timing->screen_read_ms : nullptr, stage_start);
    stage_start = InputClock::now();
    if (scalar) {
      CopyScalarTopOrigin(rgba, screen_w, screen_h, channel, &out->buffers[index]);
    } else {
      CopyRgbTopOrigin(rgba, screen_w, screen_h, &out->buffers[index]);
    }
    AddInputElapsed(timing ? &timing->screen_resize_ms : nullptr, stage_start);
    return true;
  };

  InputTimePoint stage_start = InputClock::now();
  if (!ReadTextureResizedRGBA32F(gbuffer.Texture(GBufferAttachment::kPosition),
                                 gbuffer.width(), gbuffer.height(),
                                 screen_w, screen_h, GL_LINEAR,
                                 config.use_pbo_readback, &position_rgba)) {
    if (error) {
      *error = "Failed to stage Position attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  CopyRgbTopOrigin(position_rgba, screen_w, screen_h, &out->buffers[0]);
  CopyRgbaTopOrigin(position_rgba, screen_w, screen_h, &position_top_rgba);
  AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);

  if (!read_gbuffer(GBufferAttachment::kNormal, 1, false) ||
      !read_gbuffer(GBufferAttachment::kOutDir, 2, false) ||
      !read_gbuffer(GBufferAttachment::kDiffuse, 3, false) ||
      !read_gbuffer(GBufferAttachment::kReflect, 4, false)) {
    if (error) {
      *error = "Failed to stage G-buffer RGB attachment.";
    }
    return false;
  }

  stage_start = InputClock::now();
  if (!ReadTextureResizedRGBA32F(gbuffer.Texture(GBufferAttachment::kGloss),
                                 gbuffer.width(), gbuffer.height(),
                                 screen_w, screen_h, GL_LINEAR,
                                 config.use_pbo_readback, &rgba)) {
    if (error) {
      *error = "Failed to stage Gloss attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->gbuffer_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  RoughnessFromGlossSameSizeTopOrigin(rgba, screen_w, screen_h, position_top_rgba,
                                      &out->buffers[5]);
  DepthFromPositionSameSizeTopOrigin(position_top_rgba, screen_w, screen_h,
                                     camera_pos, &out->buffers[9]);
  AddInputElapsed(timing ? &timing->gbuffer_resize_ms : nullptr, stage_start);

  if (!read_screen(ScreenSpaceLightAttachment::kSSLightVec, 6, false)) {
    if (error) {
      *error = "Failed to stage screen-space light vector attachment.";
    }
    return false;
  }

  stage_start = InputClock::now();
  if (!ReadTextureResizedRGBA32F(screen_space_light.Texture(
                                     ScreenSpaceLightAttachment::kSSLightDepth),
                                 screen_space_light.width(),
                                 screen_space_light.height(), screen_w, screen_h,
                                 GL_LINEAR, config.use_pbo_readback, &rgba)) {
    if (error) {
      *error = "Failed to stage screen-space light depth attachment.";
    }
    return false;
  }
  AddInputElapsed(timing ? &timing->screen_read_ms : nullptr, stage_start);
  stage_start = InputClock::now();
  CopyScalarTopOrigin(rgba, screen_w, screen_h, 1, &out->buffers[7]);
  CopyScalarTopOrigin(rgba, screen_w, screen_h, 0, &out->buffers[8]);
  AddInputElapsed(timing ? &timing->screen_resize_ms : nullptr, stage_start);

  std::vector<float> light_position_stage_rgba;
  std::vector<float> light_normal_stage_rgba;
  std::vector<float> light_albedo_stage_rgba;
  std::vector<float> light_position_rgba;
  std::vector<float> light_normal_rgba;
  std::vector<float> light_albedo_rgba;

  auto read_light = [&](LightSpaceAttachment attachment,
                        std::vector<float>* stage_rgba,
                        std::vector<float>* top_rgba,
                        const char* label) -> bool {
    InputTimePoint read_start = InputClock::now();
    if (!ReadTextureResizedRGBA32F(screen_space_light.Texture(attachment),
                                   screen_space_light.light_space_width(),
                                   screen_space_light.light_space_height(),
                                   rsm_w, rsm_h, GL_LINEAR,
                                   config.use_pbo_readback, stage_rgba)) {
      if (error) {
        *error = std::string("Failed to stage ") + label + " attachment.";
      }
      return false;
    }
    AddInputElapsed(timing ? &timing->light_read_ms : nullptr, read_start);
    InputTimePoint convert_start = InputClock::now();
    CopyRgbaTopOrigin(*stage_rgba, rsm_w, rsm_h, top_rgba);
    AddInputElapsed(timing ? &timing->light_downsample_ms : nullptr, convert_start);
    return true;
  };

  if (!read_light(LightSpaceAttachment::kSMPosition, &light_position_stage_rgba,
                  &light_position_rgba, "smPosition") ||
      !read_light(LightSpaceAttachment::kSMNormal, &light_normal_stage_rgba,
                  &light_normal_rgba, "smNormal") ||
      !read_light(LightSpaceAttachment::kSMFlux, &light_albedo_stage_rgba,
                  &light_albedo_rgba, "smFlux")) {
    return false;
  }

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
  if (config.use_gpu_staging) {
    return BuildRuntimeInputBuffersGpuStaging(gbuffer, screen_space_light, camera_pos,
                                             config, out, error, timing);
  }
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
