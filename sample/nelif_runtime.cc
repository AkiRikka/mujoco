// NeLiF runtime scaffold sample.
//
// This sample runs the current MuJoCo-side NeLiF render passes every frame and
// displays the runtime composition output. Direct/shadow are analytic baselines
// matching direct_from_raw.py; indirect and shadow can be supplied by the
// optional ONNX Runtime backend when the sample is built with the ONNX Runtime
// C/C++ SDK.

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include "experimental/nelif/exr_writer.h"
#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/onnx_backend.h"
#include "experimental/nelif/runtime_pass.h"
#include "experimental/nelif/screen_space_light_pass.h"
#include "nelif_decoder_plugins.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <vector>

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

enum class ViewMode : int {
  kMujoco = 0,
  kRuntimeShading = 1,
  kDirectUnshadowed = 2,
  kDirectShadowed = 3,
  kShadow = 4,
  kIndirect = 5,
  kPosition = 6,
  kNormal = 7,
  kSSLightDepth = 8,
  kSSLightVec = 9,
};

mjModel* m = nullptr;
mjData* d = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

mujoco::nelif::GBufferPass gbuffer;
mujoco::nelif::ScreenSpaceLightPass screen_space_light;
mujoco::nelif::RuntimePass runtime_pass;
mujoco::nelif::RuntimeConfig runtime_config;
mujoco::nelif::OnnxIndirectBackend indirect_backend;
mujoco::nelif::OnnxIndirectConfig indirect_config;
mujoco::nelif::OnnxShadowBackend shadow_backend;
mujoco::nelif::OnnxShadowConfig shadow_config;

ViewMode view_mode = ViewMode::kRuntimeShading;
std::string model_path;
int window_width = 1200;
int window_height = 900;
int face_size = 512;
GLuint indirect_texture = 0;
GLuint shadow_texture = 0;
bool dump_requested = false;
bool dump_first_frame = false;
bool exit_after_dump = false;
bool dump_completed = false;
bool indirect_backend_warned = false;
bool shadow_backend_warned = false;
bool runtime_input_warned = false;
bool indirect_init_failed = false;
bool shadow_init_failed = false;
bool indirect_inference_failed = false;
bool shadow_inference_failed = false;
bool runtime_input_failed = false;
bool profile_runtime = false;
int profile_warmup_frames = 0;
int exit_after_frames = 0;
int rendered_frames = 0;
int dump_index = 0;
std::optional<std::string> camera_name;
std::optional<std::string> key_name;
std::optional<std::string> indirect_onnx_path;
std::optional<std::string> shadow_onnx_path;
std::optional<std::filesystem::path> output_root;
std::optional<std::filesystem::path> profile_output;
bool onnx_gpu_staging = false;

double ElapsedMs(TimePoint start, TimePoint end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct FrameTiming {
  double sim_ms = 0.0;
  double scene_ms = 0.0;
  double gbuffer_ms = 0.0;
  double light_ms = 0.0;
  double input_pack_ms = 0.0;
  double input_gbuffer_read_ms = 0.0;
  double input_gbuffer_resize_ms = 0.0;
  double input_screen_read_ms = 0.0;
  double input_screen_resize_ms = 0.0;
  double input_light_read_ms = 0.0;
  double input_light_downsample_ms = 0.0;
  double input_light_mask_ms = 0.0;
  double input_finalize_ms = 0.0;
  double indirect_onnx_ms = 0.0;
  double shadow_onnx_ms = 0.0;
  double upload_ms = 0.0;
  double compose_ms = 0.0;
  double draw_ms = 0.0;
  double swap_poll_ms = 0.0;
  double total_ms = 0.0;
};

struct RuntimeProfiler {
  FrameTiming rolling_totals;
  FrameTiming final_totals;
  int rolling_frames = 0;
  int profiled_frames = 0;
  int total_frames = 0;
  int warmup_frames = 0;
  TimePoint last_print = Clock::now();

  void SetWarmupFrames(int warmup) {
    warmup_frames = warmup;
  }

  static void AddTiming(FrameTiming* total, const FrameTiming& timing) {
    total->sim_ms += timing.sim_ms;
    total->scene_ms += timing.scene_ms;
    total->gbuffer_ms += timing.gbuffer_ms;
    total->light_ms += timing.light_ms;
    total->input_pack_ms += timing.input_pack_ms;
    total->input_gbuffer_read_ms += timing.input_gbuffer_read_ms;
    total->input_gbuffer_resize_ms += timing.input_gbuffer_resize_ms;
    total->input_screen_read_ms += timing.input_screen_read_ms;
    total->input_screen_resize_ms += timing.input_screen_resize_ms;
    total->input_light_read_ms += timing.input_light_read_ms;
    total->input_light_downsample_ms += timing.input_light_downsample_ms;
    total->input_light_mask_ms += timing.input_light_mask_ms;
    total->input_finalize_ms += timing.input_finalize_ms;
    total->indirect_onnx_ms += timing.indirect_onnx_ms;
    total->shadow_onnx_ms += timing.shadow_onnx_ms;
    total->upload_ms += timing.upload_ms;
    total->compose_ms += timing.compose_ms;
    total->draw_ms += timing.draw_ms;
    total->swap_poll_ms += timing.swap_poll_ms;
    total->total_ms += timing.total_ms;
  }

  void Add(const FrameTiming& timing) {
    ++total_frames;
    if (total_frames <= warmup_frames) {
      return;
    }
    AddTiming(&rolling_totals, timing);
    AddTiming(&final_totals, timing);
    ++rolling_frames;
    ++profiled_frames;
  }

  void MaybePrint() {
    const TimePoint now = Clock::now();
    if (ElapsedMs(last_print, now) < 1000.0 || rolling_frames <= 0) {
      return;
    }

    const double inv = 1.0 / static_cast<double>(rolling_frames);
    const double avg_total = rolling_totals.total_ms * inv;
    const double fps = avg_total > 1e-6 ? 1000.0 / avg_total : 0.0;
    std::printf(
        "[nelif_profile] frames=%d fps=%.2f total=%.3fms sim=%.3f scene=%.3f "
        "gbuffer=%.3f light=%.3f input_pack=%.3f indirect_onnx=%.3f "
        "shadow_onnx=%.3f upload=%.3f compose=%.3f draw=%.3f swap_poll=%.3f\n",
        rolling_frames, fps, avg_total, rolling_totals.sim_ms * inv,
        rolling_totals.scene_ms * inv, rolling_totals.gbuffer_ms * inv,
        rolling_totals.light_ms * inv, rolling_totals.input_pack_ms * inv,
        rolling_totals.indirect_onnx_ms * inv, rolling_totals.shadow_onnx_ms * inv,
        rolling_totals.upload_ms * inv, rolling_totals.compose_ms * inv,
        rolling_totals.draw_ms * inv, rolling_totals.swap_poll_ms * inv);
    std::fflush(stdout);
    rolling_totals = FrameTiming();
    rolling_frames = 0;
    last_print = now;
  }

  FrameTiming Average() const {
    FrameTiming average;
    if (profiled_frames <= 0) {
      return average;
    }
    const double inv = 1.0 / static_cast<double>(profiled_frames);
    average.sim_ms = final_totals.sim_ms * inv;
    average.scene_ms = final_totals.scene_ms * inv;
    average.gbuffer_ms = final_totals.gbuffer_ms * inv;
    average.light_ms = final_totals.light_ms * inv;
    average.input_pack_ms = final_totals.input_pack_ms * inv;
    average.input_gbuffer_read_ms = final_totals.input_gbuffer_read_ms * inv;
    average.input_gbuffer_resize_ms = final_totals.input_gbuffer_resize_ms * inv;
    average.input_screen_read_ms = final_totals.input_screen_read_ms * inv;
    average.input_screen_resize_ms = final_totals.input_screen_resize_ms * inv;
    average.input_light_read_ms = final_totals.input_light_read_ms * inv;
    average.input_light_downsample_ms = final_totals.input_light_downsample_ms * inv;
    average.input_light_mask_ms = final_totals.input_light_mask_ms * inv;
    average.input_finalize_ms = final_totals.input_finalize_ms * inv;
    average.indirect_onnx_ms = final_totals.indirect_onnx_ms * inv;
    average.shadow_onnx_ms = final_totals.shadow_onnx_ms * inv;
    average.upload_ms = final_totals.upload_ms * inv;
    average.compose_ms = final_totals.compose_ms * inv;
    average.draw_ms = final_totals.draw_ms * inv;
    average.swap_poll_ms = final_totals.swap_poll_ms * inv;
    average.total_ms = final_totals.total_ms * inv;
    return average;
  }

  double Fps() const {
    const FrameTiming average = Average();
    return average.total_ms > 1e-6 ? 1000.0 / average.total_ms : 0.0;
  }
};

RuntimeProfiler profiler;

bool ParseIntFlagValue(const char* value, int* out) {
  if (!value || !out) {
    return false;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed > (1 << 20)) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

bool ParseNonnegativeIntFlagValue(const char* value, int* out) {
  if (!value || !out) {
    return false;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed < 0 || parsed > (1 << 20)) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

bool ParseFloatFlagValue(const char* value, float* out) {
  if (!value || !out) {
    return false;
  }
  char* end = nullptr;
  float parsed = std::strtof(value, &end);
  if (!end || *end != '\0') {
    return false;
  }
  *out = parsed;
  return true;
}

void PrintUsage() {
  std::printf(
      "USAGE: nelif_runtime modelfile [--width px] [--height px] [--face-size px] "
      "[--camera name] [--key name] [--shadow-bias v] [--soft-shadow-radius v] "
      "[--diffuse-scale v] [--specular-scale v] [--exposure v] "
      "[--indirect-onnx path] [--shadow-onnx path] "
      "[--onnx-provider cpu|coreml|cuda|tensorrt] "
      "[--onnx-input-staging cpu|gpu] "
      "[--onnx-screen-size px] [--onnx-screen-width px] "
      "[--onnx-screen-height px] [--onnx-rsm-face-size px] [--onnx-max-scale r g b] "
      "[--profile-runtime] [--profile-warmup-frames n] [--profile-output path] "
      "[--exit-after-frames n] [--dump-first-frame] [--exit-after-dump] "
      "[--output-dir dir]\n");
}

bool ParseArgs(int argc, const char** argv) {
  if (argc < 2) {
    return false;
  }
  model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--width")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &window_width)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--height")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &window_height)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--face-size")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &face_size)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--camera")) {
      if (i + 1 >= argc) {
        return false;
      }
      camera_name = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--key")) {
      if (i + 1 >= argc) {
        return false;
      }
      key_name = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--shadow-bias")) {
      if (i + 1 >= argc || !ParseFloatFlagValue(argv[++i], &runtime_config.shadow_bias)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--soft-shadow-radius")) {
      if (i + 1 >= argc ||
          !ParseFloatFlagValue(argv[++i], &runtime_config.soft_shadow_radius)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--diffuse-scale")) {
      if (i + 1 >= argc || !ParseFloatFlagValue(argv[++i], &runtime_config.diffuse_scale)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--specular-scale")) {
      if (i + 1 >= argc || !ParseFloatFlagValue(argv[++i], &runtime_config.specular_scale)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--exposure")) {
      if (i + 1 >= argc ||
          !ParseFloatFlagValue(argv[++i], &runtime_config.display_exposure)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--indirect-onnx")) {
      if (i + 1 >= argc) {
        return false;
      }
      indirect_onnx_path = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--shadow-onnx")) {
      if (i + 1 >= argc) {
        return false;
      }
      shadow_onnx_path = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--onnx-provider")) {
      if (i + 1 >= argc) {
        return false;
      }
      const std::string provider = std::string(argv[++i]);
      indirect_config.execution_provider = provider;
      shadow_config.execution_provider = provider;
    } else if (!std::strcmp(argv[i], "--onnx-input-staging")) {
      if (i + 1 >= argc) {
        return false;
      }
      const char* value = argv[++i];
      if (!std::strcmp(value, "cpu")) {
        onnx_gpu_staging = false;
      } else if (!std::strcmp(value, "gpu")) {
        onnx_gpu_staging = true;
      } else {
        std::fprintf(stderr, "Invalid --onnx-input-staging value: %s\n", value);
        return false;
      }
    } else if (!std::strcmp(argv[i], "--onnx-screen-size")) {
      int size = 0;
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &size)) {
        return false;
      }
      indirect_config.screen_width = size;
      indirect_config.screen_height = size;
      shadow_config.screen_width = size;
      shadow_config.screen_height = size;
    } else if (!std::strcmp(argv[i], "--onnx-screen-width")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &indirect_config.screen_width)) {
        return false;
      }
      shadow_config.screen_width = indirect_config.screen_width;
    } else if (!std::strcmp(argv[i], "--onnx-screen-height")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &indirect_config.screen_height)) {
        return false;
      }
      shadow_config.screen_height = indirect_config.screen_height;
    } else if (!std::strcmp(argv[i], "--onnx-rsm-face-size")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &indirect_config.rsm_face_size)) {
        return false;
      }
      shadow_config.rsm_face_size = indirect_config.rsm_face_size;
    } else if (!std::strcmp(argv[i], "--onnx-max-scale")) {
      if (i + 3 >= argc ||
          !ParseFloatFlagValue(argv[++i], &indirect_config.max_scale[0]) ||
          !ParseFloatFlagValue(argv[++i], &indirect_config.max_scale[1]) ||
          !ParseFloatFlagValue(argv[++i], &indirect_config.max_scale[2])) {
        return false;
      }
      shadow_config.max_scale[0] = indirect_config.max_scale[0];
      shadow_config.max_scale[1] = indirect_config.max_scale[1];
      shadow_config.max_scale[2] = indirect_config.max_scale[2];
    } else if (!std::strcmp(argv[i], "--dump-first-frame")) {
      dump_first_frame = true;
    } else if (!std::strcmp(argv[i], "--exit-after-dump")) {
      exit_after_dump = true;
    } else if (!std::strcmp(argv[i], "--profile-runtime")) {
      profile_runtime = true;
    } else if (!std::strcmp(argv[i], "--profile-warmup-frames")) {
      if (i + 1 >= argc || !ParseNonnegativeIntFlagValue(argv[++i],
                                                         &profile_warmup_frames)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--profile-output")) {
      if (i + 1 >= argc) {
        return false;
      }
      profile_output = std::filesystem::path(argv[++i]);
      profile_runtime = true;
    } else if (!std::strcmp(argv[i], "--exit-after-frames")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &exit_after_frames)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--output-dir")) {
      if (i + 1 >= argc) {
        return false;
      }
      output_root = std::filesystem::path(argv[++i]);
    } else {
      return false;
    }
  }
  indirect_config.use_gpu_staging = onnx_gpu_staging;
  shadow_config.use_gpu_staging = onnx_gpu_staging;
  return true;
}

std::filesystem::path DefaultOutputRoot() {
  std::filesystem::path current = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::exists(current / "tmp-res") &&
        std::filesystem::exists(current / "mujoco")) {
      return current / "tmp-res" / "nelif_runtime_dump";
    }
    if (current == current.root_path()) {
      return std::filesystem::current_path() / "nelif_runtime_dump";
    }
    current = current.parent_path();
  }
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

const char* JsonBool(bool value) {
  return value ? "true" : "false";
}

bool WriteProfileJson(const std::filesystem::path& path, int framebuffer_width,
                      int framebuffer_height) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
      std::fprintf(stderr, "Failed to create profile output dir %s: %s\n",
                   parent.string().c_str(), error.message().c_str());
      return false;
    }
  }

  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open profile output %s\n", path.string().c_str());
    return false;
  }

  const FrameTiming average = profiler.Average();
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"schema\": \"nelif.runtime_profile.v1\",\n";
  out << "  \"model_file\": \"" << JsonEscape(model_path) << "\",\n";
  out << "  \"provider\": \"" << JsonEscape(indirect_config.execution_provider) << "\",\n";
  out << "  \"window_size\": [" << window_width << ", " << window_height << "],\n";
  out << "  \"framebuffer_size\": [" << framebuffer_width << ", " << framebuffer_height << "],\n";
  out << "  \"face_size\": " << face_size << ",\n";
  out << "  \"onnx_screen_size\": [" << indirect_config.screen_width << ", "
      << indirect_config.screen_height << "],\n";
  out << "  \"onnx_rsm_face_size\": " << indirect_config.rsm_face_size << ",\n";
  out << "  \"onnx_input_staging\": \""
      << (onnx_gpu_staging ? "gpu" : "cpu") << "\",\n";
  out << "  \"indirect_onnx\": \"" << JsonEscape(indirect_onnx_path.value_or("")) << "\",\n";
  out << "  \"shadow_onnx\": \"" << JsonEscape(shadow_onnx_path.value_or("")) << "\",\n";
  out << "  \"indirect_enabled\": " << JsonBool(indirect_backend.IsEnabled()) << ",\n";
  out << "  \"shadow_enabled\": " << JsonBool(shadow_backend.IsEnabled()) << ",\n";
  out << "  \"indirect_init_failed\": " << JsonBool(indirect_init_failed) << ",\n";
  out << "  \"shadow_init_failed\": " << JsonBool(shadow_init_failed) << ",\n";
  out << "  \"runtime_input_failed\": " << JsonBool(runtime_input_failed) << ",\n";
  out << "  \"indirect_inference_failed\": " << JsonBool(indirect_inference_failed) << ",\n";
  out << "  \"shadow_inference_failed\": " << JsonBool(shadow_inference_failed) << ",\n";
  out << "  \"frames_total\": " << profiler.total_frames << ",\n";
  out << "  \"warmup_frames\": " << profiler.warmup_frames << ",\n";
  out << "  \"profiled_frames\": " << profiler.profiled_frames << ",\n";
  out << "  \"fps\": " << profiler.Fps() << ",\n";
  out << "  \"avg_ms\": {\n";
  out << "    \"total\": " << average.total_ms << ",\n";
  out << "    \"sim\": " << average.sim_ms << ",\n";
  out << "    \"scene\": " << average.scene_ms << ",\n";
  out << "    \"gbuffer\": " << average.gbuffer_ms << ",\n";
  out << "    \"light\": " << average.light_ms << ",\n";
  out << "    \"input_pack\": " << average.input_pack_ms << ",\n";
  out << "    \"indirect_onnx\": " << average.indirect_onnx_ms << ",\n";
  out << "    \"shadow_onnx\": " << average.shadow_onnx_ms << ",\n";
  out << "    \"upload\": " << average.upload_ms << ",\n";
  out << "    \"compose\": " << average.compose_ms << ",\n";
  out << "    \"draw\": " << average.draw_ms << ",\n";
  out << "    \"swap_poll\": " << average.swap_poll_ms << "\n";
  out << "  },\n";
  out << "  \"input_pack_breakdown_ms\": {\n";
  out << "    \"gbuffer_read\": " << average.input_gbuffer_read_ms << ",\n";
  out << "    \"gbuffer_resize\": " << average.input_gbuffer_resize_ms << ",\n";
  out << "    \"screen_read\": " << average.input_screen_read_ms << ",\n";
  out << "    \"screen_resize\": " << average.input_screen_resize_ms << ",\n";
  out << "    \"light_read\": " << average.input_light_read_ms << ",\n";
  out << "    \"light_downsample\": " << average.input_light_downsample_ms << ",\n";
  out << "    \"light_mask\": " << average.input_light_mask_ms << ",\n";
  out << "    \"finalize\": " << average.input_finalize_ms << "\n";
  out << "  }\n";
  out << "}\n";
  if (!out) {
    std::fprintf(stderr, "Failed to write profile output %s\n", path.string().c_str());
    return false;
  }
  std::printf("Wrote NeLiF runtime profile to %s\n", path.string().c_str());
  return true;
}

void UploadRuntimeTexture(GLuint* texture, const std::vector<float>& rgba, int width, int height) {
  if (rgba.empty() || width <= 0 || height <= 0) {
    return;
  }
  if (!*texture) {
    glGenTextures(1, texture);
  }
  glBindTexture(GL_TEXTURE_2D, *texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT,
               rgba.data());
  glBindTexture(GL_TEXTURE_2D, 0);
}

void CurrentCameraPos(float camera_pos[3]) {
  const mjvGLCamera render_camera = mjv_averageCamera(scn.camera, scn.camera + 1);
  camera_pos[0] = static_cast<float>(render_camera.pos[0]);
  camera_pos[1] = static_cast<float>(render_camera.pos[1]);
  camera_pos[2] = static_cast<float>(render_camera.pos[2]);
}

GLuint RunIndirectBackend(mujoco::nelif::OnnxRuntimeInputs* inputs,
                          double* inference_ms, double* upload_ms) {
  if (inference_ms) {
    *inference_ms = 0.0;
  }
  if (upload_ms) {
    *upload_ms = 0.0;
  }
  if (!indirect_backend.IsEnabled()) {
    return 0;
  }
  if (!inputs) {
    indirect_inference_failed = true;
    if (!indirect_backend_warned) {
      std::fprintf(stderr, "ONNX indirect inference skipped: runtime inputs unavailable.\n");
      indirect_backend_warned = true;
    }
    return 0;
  }

  std::vector<float> indirect_rgba;
  std::string error;
  TimePoint start = Clock::now();
  if (!indirect_backend.Run(*inputs, &indirect_rgba, &error)) {
    indirect_inference_failed = true;
    if (inference_ms) {
      *inference_ms = ElapsedMs(start, Clock::now());
    }
    if (!indirect_backend_warned) {
      std::fprintf(stderr, "ONNX indirect inference failed: %s\n", error.c_str());
      indirect_backend_warned = true;
    }
    return 0;
  }
  if (inference_ms) {
    *inference_ms = ElapsedMs(start, Clock::now());
  }

  start = Clock::now();
  UploadRuntimeTexture(&indirect_texture, indirect_rgba, indirect_backend.output_width(),
                       indirect_backend.output_height());
  if (upload_ms) {
    *upload_ms = ElapsedMs(start, Clock::now());
  }
  return indirect_texture;
}

GLuint RunShadowBackend(mujoco::nelif::OnnxRuntimeInputs* inputs,
                        double* inference_ms, double* upload_ms) {
  if (inference_ms) {
    *inference_ms = 0.0;
  }
  if (upload_ms) {
    *upload_ms = 0.0;
  }
  if (!shadow_backend.IsEnabled()) {
    return 0;
  }
  if (!inputs) {
    shadow_inference_failed = true;
    if (!shadow_backend_warned) {
      std::fprintf(stderr, "ONNX shadow inference skipped: runtime inputs unavailable.\n");
      shadow_backend_warned = true;
    }
    return 0;
  }

  std::vector<float> shadow_rgba;
  std::string error;
  TimePoint start = Clock::now();
  if (!shadow_backend.Run(*inputs, &shadow_rgba, &error)) {
    shadow_inference_failed = true;
    if (inference_ms) {
      *inference_ms = ElapsedMs(start, Clock::now());
    }
    if (!shadow_backend_warned) {
      std::fprintf(stderr, "ONNX shadow inference failed: %s\n", error.c_str());
      shadow_backend_warned = true;
    }
    return 0;
  }
  if (inference_ms) {
    *inference_ms = ElapsedMs(start, Clock::now());
  }

  start = Clock::now();
  UploadRuntimeTexture(&shadow_texture, shadow_rgba, shadow_backend.output_width(),
                       shadow_backend.output_height());
  if (upload_ms) {
    *upload_ms = ElapsedMs(start, Clock::now());
  }
  return shadow_texture;
}

bool DumpAttachment(const std::filesystem::path& dir, const char* name,
                    mujoco::nelif::RuntimeAttachment attachment) {
  std::vector<float> rgba;
  if (!runtime_pass.Readback(attachment, &rgba)) {
    std::printf("Failed to read back %s\n", name);
    return false;
  }
  std::string error;
  if (!mujoco::nelif::SaveRgba32fExr((dir / (std::string(name) + ".exr")).string(),
                                     runtime_pass.width(), runtime_pass.height(), rgba,
                                     &error, /*flip_y=*/true)) {
    std::printf("%s\n", error.c_str());
    return false;
  }
  return true;
}

void DumpCurrentFrame() {
  std::filesystem::path root = output_root.value_or(DefaultOutputRoot());
  char frame_name[32];
  std::snprintf(frame_name, sizeof(frame_name), "frame_%06d", dump_index++);
  std::filesystem::path dir = root / frame_name;
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  if (error) {
    std::printf("Failed to create dump dir %s: %s\n", dir.string().c_str(),
                error.message().c_str());
    return;
  }

  DumpAttachment(dir, "RuntimeDirectUnshadowed",
                 mujoco::nelif::RuntimeAttachment::kDirectUnshadowed);
  DumpAttachment(dir, "RuntimeDirectShadowed",
                 mujoco::nelif::RuntimeAttachment::kDirectShadowed);
  DumpAttachment(dir, indirect_backend.IsEnabled() ? "PredIndirectShading" : "PredIndirectStub",
                 mujoco::nelif::RuntimeAttachment::kIndirect);
  DumpAttachment(dir, "PredShadow", mujoco::nelif::RuntimeAttachment::kShadow);
  DumpAttachment(dir, "RuntimeShading", mujoco::nelif::RuntimeAttachment::kComposed);
  std::printf("Dumped NeLiF runtime outputs to %s\n", dir.string().c_str());
}

bool LoadModel(const char* path) {
  char error[1000] = "Could not load model";
  if (std::strlen(path) > 4 && !std::strcmp(path + std::strlen(path) - 4, ".mjb")) {
    m = mj_loadModel(path, nullptr);
  } else {
    m = mj_loadXML(path, nullptr, error, sizeof(error));
  }
  if (!m) {
    std::fprintf(stderr, "Load model error: %s\n", error);
    return false;
  }
  d = mj_makeData(m);
  mj_forward(m, d);
  return true;
}

bool ResetToKeyframe() {
  if (!key_name.has_value()) {
    return true;
  }
  int keyid = mj_name2id(m, mjOBJ_KEY, key_name->c_str());
  if (keyid < 0) {
    std::fprintf(stderr, "Could not find keyframe '%s'\n", key_name->c_str());
    return false;
  }
  mj_resetDataKeyframe(m, d, keyid);
  mj_forward(m, d);
  return true;
}

bool InitCamera() {
  mjv_defaultCamera(&cam);
  if (camera_name.has_value()) {
    int camid = mj_name2id(m, mjOBJ_CAMERA, camera_name->c_str());
    if (camid < 0) {
      std::fprintf(stderr, "Could not find camera '%s'\n", camera_name->c_str());
      return false;
    }
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = camid;
    return true;
  }
  int camid = mj_name2id(m, mjOBJ_CAMERA, "nelif_fixed_cam");
  if (camid >= 0) {
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = camid;
  } else {
    mjv_defaultFreeCamera(m, &cam);
  }
  return true;
}

void Keyboard(GLFWwindow* window, int key, int, int act, int) {
  if (act != GLFW_PRESS) {
    return;
  }
  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, 1);
  } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
    view_mode = static_cast<ViewMode>(key - GLFW_KEY_0);
    std::printf("View mode %d\n", static_cast<int>(view_mode));
  } else if (key == GLFW_KEY_D) {
    dump_requested = true;
  } else if (key == GLFW_KEY_LEFT_BRACKET) {
    runtime_config.display_exposure *= 0.5f;
    std::printf("Exposure %.3f\n", runtime_config.display_exposure);
  } else if (key == GLFW_KEY_RIGHT_BRACKET) {
    runtime_config.display_exposure *= 2.0f;
    std::printf("Exposure %.3f\n", runtime_config.display_exposure);
  } else if (key == GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

void DrawCurrentView(const mjrRect& viewport) {
  switch (view_mode) {
    case ViewMode::kMujoco:
      mjr_render(viewport, &scn, &con);
      break;
    case ViewMode::kRuntimeShading:
      runtime_pass.DrawDebug(mujoco::nelif::RuntimeAttachment::kComposed, viewport.width,
                             viewport.height, runtime_config.display_exposure);
      break;
    case ViewMode::kDirectUnshadowed:
      runtime_pass.DrawDebug(mujoco::nelif::RuntimeAttachment::kDirectUnshadowed,
                             viewport.width, viewport.height,
                             runtime_config.display_exposure);
      break;
    case ViewMode::kDirectShadowed:
      runtime_pass.DrawDebug(mujoco::nelif::RuntimeAttachment::kDirectShadowed,
                             viewport.width, viewport.height,
                             runtime_config.display_exposure);
      break;
    case ViewMode::kShadow:
      runtime_pass.DrawDebug(mujoco::nelif::RuntimeAttachment::kShadow, viewport.width,
                             viewport.height, 1.0f);
      break;
    case ViewMode::kIndirect:
      runtime_pass.DrawDebug(mujoco::nelif::RuntimeAttachment::kIndirect, viewport.width,
                             viewport.height, runtime_config.display_exposure);
      break;
    case ViewMode::kPosition:
      gbuffer.DrawDebug(mujoco::nelif::kDebugPosition, viewport.width, viewport.height);
      break;
    case ViewMode::kNormal:
      gbuffer.DrawDebug(mujoco::nelif::kDebugNormal, viewport.width, viewport.height);
      break;
    case ViewMode::kSSLightDepth:
      screen_space_light.DrawDebug(mujoco::nelif::kDebugSSLightDepth, viewport.width,
                                   viewport.height);
      break;
    case ViewMode::kSSLightVec:
      screen_space_light.DrawDebug(mujoco::nelif::kDebugSSLightVec, viewport.width,
                                   viewport.height);
      break;
  }
}

}  // namespace

int main(int argc, const char** argv) {
  if (!ParseArgs(argc, argv)) {
    PrintUsage();
    return EXIT_FAILURE;
  }

  nelif_sample::LoadMeshDecoderPlugins(argv[0]);
  if (!LoadModel(argv[1]) || !ResetToKeyframe()) {
    return EXIT_FAILURE;
  }

  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif
  GLFWwindow* window =
      glfwCreateWindow(window_width, window_height, "NeLiF Runtime", nullptr, nullptr);
  if (!window) {
    mju_error("Could not create GLFW window");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetKeyCallback(window, Keyboard);

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
  if (!InitCamera()) {
    return EXIT_FAILURE;
  }

  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
  gbuffer.Init(framebuffer_width, framebuffer_height);
  screen_space_light.Init(framebuffer_width, framebuffer_height, face_size);
  runtime_pass.Init(framebuffer_width, framebuffer_height);
  if (indirect_onnx_path.has_value()) {
    indirect_config.model_path = *indirect_onnx_path;
    std::string error;
    if (indirect_backend.Init(indirect_config, &error)) {
      std::printf("%s: %s\n", indirect_backend.Status(), indirect_config.model_path.c_str());
    } else {
      indirect_init_failed = true;
      std::fprintf(stderr, "ONNX indirect backend disabled: %s\n", error.c_str());
    }
  }
  if (shadow_onnx_path.has_value()) {
    shadow_config.model_path = *shadow_onnx_path;
    std::string error;
    if (shadow_backend.Init(shadow_config, &error)) {
      std::printf("%s: %s\n", shadow_backend.Status(), shadow_config.model_path.c_str());
    } else {
      shadow_init_failed = true;
      std::fprintf(stderr, "ONNX shadow backend disabled: %s\n", error.c_str());
    }
  }

  std::printf(
      "View keys: 0 MuJoCo, 1 RuntimeShading, 2 DirectUnshadowed, 3 DirectShadowed, "
      "4 Shadow, 5 Indirect, 6 Position, 7 Normal, 8 SSLightDepth, 9 SSLightVec, "
      "[/] exposure, D dump\n");

  profiler.SetWarmupFrames(profile_warmup_frames);
  int last_framebuffer_width = framebuffer_width;
  int last_framebuffer_height = framebuffer_height;

  while (!glfwWindowShouldClose(window)) {
    FrameTiming timing;
    const TimePoint frame_start = Clock::now();
    TimePoint stage_start = frame_start;

    mj_step(m, d);
    timing.sim_ms = ElapsedMs(stage_start, Clock::now());

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    last_framebuffer_width = viewport.width;
    last_framebuffer_height = viewport.height;

    stage_start = Clock::now();
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    timing.scene_ms = ElapsedMs(stage_start, Clock::now());

    stage_start = Clock::now();
    gbuffer.Resize(viewport.width, viewport.height);
    gbuffer.Render(m, &scn);
    timing.gbuffer_ms = ElapsedMs(stage_start, Clock::now());

    stage_start = Clock::now();
    screen_space_light.Resize(viewport.width, viewport.height);
    screen_space_light.Render(m, &scn, gbuffer);
    timing.light_ms = ElapsedMs(stage_start, Clock::now());

    mujoco::nelif::OnnxRuntimeInputs runtime_inputs;
    mujoco::nelif::OnnxRuntimeInputs* runtime_inputs_ptr = nullptr;
    if (indirect_backend.IsEnabled() || shadow_backend.IsEnabled()) {
      mujoco::nelif::OnnxRuntimeInputTiming input_timing;
      float camera_pos[3];
      CurrentCameraPos(camera_pos);
      const mujoco::nelif::OnnxIndirectConfig& input_config =
          indirect_backend.IsEnabled() ? indirect_config : shadow_config;
      std::string error;
      stage_start = Clock::now();
      if (mujoco::nelif::BuildOnnxRuntimeInputs(gbuffer, screen_space_light, camera_pos,
                                                input_config, &runtime_inputs, &error,
                                                &input_timing)) {
        runtime_inputs_ptr = &runtime_inputs;
        timing.input_gbuffer_read_ms = input_timing.gbuffer_read_ms;
        timing.input_gbuffer_resize_ms = input_timing.gbuffer_resize_ms;
        timing.input_screen_read_ms = input_timing.screen_read_ms;
        timing.input_screen_resize_ms = input_timing.screen_resize_ms;
        timing.input_light_read_ms = input_timing.light_read_ms;
        timing.input_light_downsample_ms = input_timing.light_downsample_ms;
        timing.input_light_mask_ms = input_timing.light_mask_ms;
        timing.input_finalize_ms = input_timing.finalize_ms;
      } else if (!runtime_input_warned) {
        runtime_input_failed = true;
        std::fprintf(stderr, "ONNX runtime input packing failed: %s\n", error.c_str());
        runtime_input_warned = true;
      }
      timing.input_pack_ms = ElapsedMs(stage_start, Clock::now());
    }

    runtime_pass.Resize(viewport.width, viewport.height);
    double indirect_upload_ms = 0.0;
    double shadow_upload_ms = 0.0;
    const GLuint neural_indirect_texture =
        RunIndirectBackend(runtime_inputs_ptr, &timing.indirect_onnx_ms,
                           &indirect_upload_ms);
    const GLuint neural_shadow_texture =
        RunShadowBackend(runtime_inputs_ptr, &timing.shadow_onnx_ms, &shadow_upload_ms);
    timing.upload_ms = indirect_upload_ms + shadow_upload_ms;

    stage_start = Clock::now();
    runtime_pass.Render(gbuffer, screen_space_light, runtime_config,
                        neural_indirect_texture, neural_shadow_texture);
    timing.compose_ms = ElapsedMs(stage_start, Clock::now());

    if (dump_requested || (dump_first_frame && !dump_completed)) {
      DumpCurrentFrame();
      dump_requested = false;
      dump_completed = true;
      if (exit_after_dump) {
        glfwSetWindowShouldClose(window, 1);
      }
    }

    stage_start = Clock::now();
    DrawCurrentView(viewport);
    timing.draw_ms = ElapsedMs(stage_start, Clock::now());

    stage_start = Clock::now();
    glfwSwapBuffers(window);
    glfwPollEvents();
    timing.swap_poll_ms = ElapsedMs(stage_start, Clock::now());
    timing.total_ms = ElapsedMs(frame_start, Clock::now());
    if (profile_runtime) {
      profiler.Add(timing);
      profiler.MaybePrint();
    }
    ++rendered_frames;
    if (exit_after_frames > 0 && rendered_frames >= exit_after_frames) {
      glfwSetWindowShouldClose(window, 1);
    }
  }

  bool profile_write_failed = false;
  if (profile_output.has_value()) {
    profile_write_failed = !WriteProfileJson(*profile_output, last_framebuffer_width,
                                             last_framebuffer_height);
  }
  const bool runtime_inference_failed =
      indirect_init_failed || shadow_init_failed || runtime_input_failed ||
      indirect_inference_failed || shadow_inference_failed;

  runtime_pass.Shutdown();
  indirect_backend.Shutdown();
  shadow_backend.Shutdown();
  if (indirect_texture) {
    glDeleteTextures(1, &indirect_texture);
    indirect_texture = 0;
  }
  if (shadow_texture) {
    glDeleteTextures(1, &shadow_texture);
    shadow_texture = 0;
  }
  screen_space_light.Shutdown();
  gbuffer.Shutdown();
  mjr_freeContext(&con);
  mjv_freeScene(&scn);
  mj_deleteData(d);
  mj_deleteModel(m);

#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return (profile_write_failed || (profile_output.has_value() && runtime_inference_failed))
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
