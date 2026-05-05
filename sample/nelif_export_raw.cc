// One-shot raw NeLiF attachment exporter.
//
// This tool renders a single MuJoCo state into the current NeLiF raw input
// contract and writes EXR files directly to a target directory. It is meant
// for offline dataset generation, not interactive debugging.

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include "experimental/nelif/exr_writer.h"
#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/screen_space_light_pass.h"
#include "nelif_decoder_plugins.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

enum class ExrPrecisionPolicy {
  kFloat = 0,
  kHalf,
  kMixed,
};

struct ExportConfig {
  std::string model_path;
  std::filesystem::path output_dir;
  std::optional<std::string> camera_name;
  std::optional<std::string> key_name;
  int width = 512;
  int height = 512;
  int face_size = 1024;
  int steps = 0;
  bool visible = false;
  bool verbose = false;
  bool write_debug = false;
  ExrPrecisionPolicy exr_precision = ExrPrecisionPolicy::kFloat;
};

void LogVerbose(const ExportConfig& config, const char* message) {
  if (config.verbose) {
    std::fprintf(stderr, "[nelif_export_raw] %s\n", message);
  }
}

std::string EscapeJson(const std::string& input) {
  std::string escaped;
  escaped.reserve(input.size());
  for (char c : input) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

bool ParseIntFlagValue(const char* value, int* out) {
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

void PrintUsage() {
  std::printf(
      "USAGE: nelif_export_raw modelfile --output-dir dir "
      "[--width px] [--height px] [--face-size px] [--steps n] "
      "[--camera name] [--key name] [--exr-precision float|half|mixed] "
      "[--visible] [--verbose] [--write-debug]\n");
}

bool ParseExrPrecision(const char* value, ExrPrecisionPolicy* out) {
  if (!value || !out) {
    return false;
  }
  if (!std::strcmp(value, "float")) {
    *out = ExrPrecisionPolicy::kFloat;
    return true;
  }
  if (!std::strcmp(value, "half")) {
    *out = ExrPrecisionPolicy::kHalf;
    return true;
  }
  if (!std::strcmp(value, "mixed")) {
    *out = ExrPrecisionPolicy::kMixed;
    return true;
  }
  return false;
}

const char* ExrPrecisionName(ExrPrecisionPolicy precision) {
  switch (precision) {
    case ExrPrecisionPolicy::kFloat:
      return "float";
    case ExrPrecisionPolicy::kHalf:
      return "half";
    case ExrPrecisionPolicy::kMixed:
      return "mixed";
  }
  return "float";
}

bool IsPositionOrDepthAttachment(const char* name) {
  return !std::strcmp(name, "Position") || !std::strcmp(name, "ssLightDepth") ||
         !std::strcmp(name, "smPosition");
}

bool SaveAttachmentAsFp16(ExrPrecisionPolicy precision, const char* name) {
  switch (precision) {
    case ExrPrecisionPolicy::kFloat:
      return false;
    case ExrPrecisionPolicy::kHalf:
      return true;
    case ExrPrecisionPolicy::kMixed:
      return !IsPositionOrDepthAttachment(name);
  }
  return false;
}

const char* StorageFormat(ExrPrecisionPolicy precision, const char* name) {
  return SaveAttachmentAsFp16(precision, name) ? "RGBA16F" : "RGBA32F";
}

bool ParseArgs(int argc, const char** argv, ExportConfig* config) {
  if (!config || argc < 2) {
    return false;
  }

  config->model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--output-dir")) {
      if (i + 1 >= argc) {
        return false;
      }
      config->output_dir = std::filesystem::path(argv[++i]);
    } else if (!std::strcmp(argv[i], "--width")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &config->width) ||
          config->width <= 0) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--height")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &config->height) ||
          config->height <= 0) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--face-size")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &config->face_size) ||
          config->face_size <= 0) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--steps")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &config->steps) ||
          config->steps < 0) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--camera")) {
      if (i + 1 >= argc) {
        return false;
      }
      config->camera_name = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--key")) {
      if (i + 1 >= argc) {
        return false;
      }
      config->key_name = std::string(argv[++i]);
    } else if (!std::strcmp(argv[i], "--exr-precision")) {
      if (i + 1 >= argc || !ParseExrPrecision(argv[++i], &config->exr_precision)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--visible")) {
      config->visible = true;
    } else if (!std::strcmp(argv[i], "--verbose")) {
      config->verbose = true;
    } else if (!std::strcmp(argv[i], "--write-debug")) {
      config->write_debug = true;
    } else {
      return false;
    }
  }

  return !config->output_dir.empty();
}

bool DumpAttachment(const std::filesystem::path& dir, const char* name,
                    int width, int height, const std::vector<float>& rgba,
                    ExrPrecisionPolicy exr_precision = ExrPrecisionPolicy::kFloat) {
  std::string error;
  if (mujoco::nelif::SaveRgba32fExr((dir / (std::string(name) + ".exr")).string(),
                                    width, height, rgba, &error, /*flip_y=*/true,
                                    SaveAttachmentAsFp16(exr_precision, name))) {
    return true;
  }
  std::fprintf(stderr, "%s\n", error.c_str());
  return false;
}

bool DumpSSLightDepthDebug(const std::filesystem::path& dir, int width, int height,
                           const std::vector<float>& ss_light_depth) {
  constexpr float kShadowDebugBias = 0.02f;
  const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (ss_light_depth.size() != pixel_count * 4) {
    std::fprintf(stderr, "ssLightDepth buffer size mismatch while writing debug outputs\n");
    return false;
  }

  std::error_code error;
  const std::filesystem::path debug_dir = dir / "_debug";
  std::filesystem::create_directories(debug_dir, error);
  if (error) {
    std::fprintf(stderr, "Failed to create debug dir %s: %s\n",
                 debug_dir.string().c_str(), error.message().c_str());
    return false;
  }

  float max_positive_delta = 0.0f;
  for (size_t i = 0; i < pixel_count; ++i) {
    const float mask = ss_light_depth[4 * i + 3];
    if (mask < 0.5f) {
      continue;
    }
    const float occluder = ss_light_depth[4 * i + 0];
    const float surface = ss_light_depth[4 * i + 1];
    max_positive_delta = std::max(max_positive_delta,
                                  std::max(surface - occluder - kShadowDebugBias, 0.0f));
  }
  if (max_positive_delta <= 1e-6f) {
    max_positive_delta = 1.0f;
  }

  std::vector<float> delta(pixel_count * 4, 0.0f);
  std::vector<float> visibility(pixel_count * 4, 0.0f);
  for (size_t i = 0; i < pixel_count; ++i) {
    const float mask = ss_light_depth[4 * i + 3];
    if (mask < 0.5f) {
      continue;
    }
    const float occluder = ss_light_depth[4 * i + 0];
    const float surface = ss_light_depth[4 * i + 1];
    const float positive_delta = std::max(surface - occluder - kShadowDebugBias, 0.0f);
    const float normalized_delta = std::min(positive_delta / max_positive_delta, 1.0f);
    const float hard_visibility = (occluder + kShadowDebugBias >= surface) ? 1.0f : 0.0f;

    delta[4 * i + 0] = normalized_delta;
    delta[4 * i + 1] = normalized_delta;
    delta[4 * i + 2] = normalized_delta;
    delta[4 * i + 3] = mask;

    visibility[4 * i + 0] = hard_visibility;
    visibility[4 * i + 1] = hard_visibility;
    visibility[4 * i + 2] = hard_visibility;
    visibility[4 * i + 3] = mask;
  }

  return DumpAttachment(debug_dir, "ssLightDepth_delta", width, height, delta) &&
         DumpAttachment(debug_dir, "ssLightDepth_visibility", width, height, visibility);
}

const char* CameraTypeName(int type) {
  switch (type) {
    case mjCAMERA_FREE:
      return "free";
    case mjCAMERA_TRACKING:
      return "tracking";
    case mjCAMERA_FIXED:
      return "fixed";
    case mjCAMERA_USER:
      return "user";
    default:
      return "unknown";
  }
}

double ComputeVerticalFovDegrees(const mjvGLCamera& camera) {
  if (camera.orthographic || camera.frustum_near <= 1e-8f) {
    return 0.0;
  }
  const double near = static_cast<double>(camera.frustum_near);
  const double top = std::atan(static_cast<double>(camera.frustum_top) / near);
  const double bottom = std::atan(static_cast<double>(-camera.frustum_bottom) / near);
  return (top + bottom) * 180.0 / 3.14159265358979323846;
}

const char* LightTypeName(mujoco::nelif::LightType type) {
  switch (type) {
    case mujoco::nelif::LightType::kPoint:
      return "point";
    case mujoco::nelif::LightType::kSphereArea:
      return "sphere_area";
    case mujoco::nelif::LightType::kRectArea:
      return "rect_area";
    case mujoco::nelif::LightType::kSpot:
      return "spot";
    default:
      return "unknown";
  }
}

void WriteManifest(const std::filesystem::path& dir, const ExportConfig& config,
                   const mjModel* model, const mjvCamera& camera,
                   const mjvScene& scene,
                   const mujoco::nelif::GBufferPass& gbuffer,
                   const mujoco::nelif::ScreenSpaceLightPass& screen_space_light) {
  std::ofstream out(dir / "manifest.json");
  if (!out) {
    std::fprintf(stderr, "Failed to open manifest for %s\n", dir.string().c_str());
    return;
  }

  const mjvGLCamera render_camera = mjv_averageCamera(scene.camera, scene.camera + 1);
  const char* resolved_camera_name = nullptr;
  if (camera.type == mjCAMERA_FIXED && camera.fixedcamid >= 0) {
    resolved_camera_name = mj_id2name(model, mjOBJ_CAMERA, camera.fixedcamid);
  }

  out << "{\n";
  out << "  \"model\": \"" << EscapeJson(config.model_path) << "\",\n";
  out << "  \"screen_width\": " << gbuffer.width() << ",\n";
  out << "  \"screen_height\": " << gbuffer.height() << ",\n";
  out << "  \"light_face_size\": " << screen_space_light.face_size() << ",\n";
  out << "  \"light_space_width\": " << screen_space_light.light_space_width() << ",\n";
  out << "  \"light_space_height\": " << screen_space_light.light_space_height() << ",\n";
  out << "  \"steps\": " << config.steps << ",\n";
  out << "  \"exr_precision\": \"" << ExrPrecisionName(config.exr_precision) << "\",\n";
  out << "  \"keyframe\": "
      << (config.key_name.has_value() ? ("\"" + EscapeJson(*config.key_name) + "\"") : "null")
      << ",\n";
  out << "  \"write_debug\": " << (config.write_debug ? "true" : "false") << ",\n";
  out << "  \"camera\": {\n";
  out << "    \"requested_name\": "
      << (config.camera_name.has_value()
              ? ("\"" + EscapeJson(*config.camera_name) + "\"")
              : "null")
      << ",\n";
  out << "    \"resolved_name\": "
      << (resolved_camera_name ? ("\"" + EscapeJson(resolved_camera_name) + "\"") : "null")
      << ",\n";
  out << "    \"type\": \"" << CameraTypeName(camera.type) << "\",\n";
  out << "    \"fixedcamid\": "
      << (camera.type == mjCAMERA_FIXED ? std::to_string(camera.fixedcamid) : "null")
      << ",\n";
  out << "    \"position\": [" << render_camera.pos[0] << ", " << render_camera.pos[1]
      << ", " << render_camera.pos[2] << "],\n";
  out << "    \"forward\": [" << render_camera.forward[0] << ", "
      << render_camera.forward[1] << ", " << render_camera.forward[2] << "],\n";
  out << "    \"up\": [" << render_camera.up[0] << ", " << render_camera.up[1] << ", "
      << render_camera.up[2] << "],\n";
  out << "    \"orthographic\": " << (render_camera.orthographic ? "true" : "false")
      << ",\n";
  out << "    \"vertical_fov_degrees\": " << ComputeVerticalFovDegrees(render_camera)
      << ",\n";
  out << "    \"frustum\": {\n";
  out << "      \"center\": " << render_camera.frustum_center << ",\n";
  out << "      \"width\": " << render_camera.frustum_width << ",\n";
  out << "      \"bottom\": " << render_camera.frustum_bottom << ",\n";
  out << "      \"top\": " << render_camera.frustum_top << ",\n";
  out << "      \"near\": " << render_camera.frustum_near << ",\n";
  out << "      \"far\": " << render_camera.frustum_far << "\n";
  out << "    }\n";
  out << "  },\n";
  out << "  \"has_active_light\": "
      << (screen_space_light.HasActiveLight() ? "true" : "false") << ",\n";
  if (screen_space_light.HasActiveLight()) {
    const mujoco::nelif::LightDescriptor& light = screen_space_light.light();
    out << "  \"light\": {\n";
    out << "    \"type\": " << static_cast<int>(light.type) << ",\n";
    out << "    \"type_name\": \"" << LightTypeName(light.type) << "\",\n";
    out << "    \"position\": [" << light.position[0] << ", " << light.position[1] << ", "
        << light.position[2] << "],\n";
    out << "    \"direction\": [" << light.direction[0] << ", " << light.direction[1] << ", "
        << light.direction[2] << "],\n";
    out << "    \"color\": [" << light.color[0] << ", " << light.color[1] << ", "
        << light.color[2] << "],\n";
    out << "    \"intensity\": " << light.intensity << ",\n";
    out << "    \"brightness\": " << light.intensity << ",\n";
    out << "    \"emission_rgb\": [" << light.color[0] * light.intensity << ", "
        << light.color[1] * light.intensity << ", "
        << light.color[2] * light.intensity << "],\n";
    out << "    \"radius\": " << light.radius << ",\n";
    out << "    \"range\": " << light.range << ",\n";
    out << "    \"cone_angle_degrees\": " << light.cone_angle_degrees << "\n";
    out << "  },\n";
  }
  out << "  \"files\": [\n";

  bool first = true;
  auto write_file_entry =
      [&](const mujoco::nelif::AttachmentContract& contract, int width, int height) {
        if (!first) {
          out << ",\n";
        }
        first = false;
        out << "    {\n";
        out << "      \"name\": \"" << contract.name << "\",\n";
        out << "      \"path\": \"" << contract.name << ".exr\",\n";
        out << "      \"width\": " << width << ",\n";
        out << "      \"height\": " << height << ",\n";
        out << "      \"file_format\": \"OpenEXR\",\n";
        out << "      \"pixel_format\": \"" << contract.format << "\",\n";
        out << "      \"storage_format\": \"" << StorageFormat(config.exr_precision, contract.name) << "\",\n";
        out << "      \"channels\": [\"" << contract.x << "\", \"" << contract.y << "\", \""
            << contract.z << "\", \"" << contract.w << "\"]\n";
        out << "    }";
      };

  for (int i = 0; i < mujoco::nelif::kGBufferAttachmentCount; ++i) {
    write_file_entry(mujoco::nelif::kGBufferContract[i], gbuffer.width(), gbuffer.height());
  }
  for (int i = 0; i < mujoco::nelif::kScreenSpaceLightAttachmentCount; ++i) {
    write_file_entry(mujoco::nelif::kScreenSpaceLightContract[i], screen_space_light.width(),
                     screen_space_light.height());
  }
  for (int i = 0; i < mujoco::nelif::kLightSpaceAttachmentCount; ++i) {
    write_file_entry(mujoco::nelif::kLightSpaceContract[i],
                     screen_space_light.light_space_width(),
                     screen_space_light.light_space_height());
  }

  out << "\n  ]\n";
  out << "}\n";
}

bool ExportRawFrame(const std::filesystem::path& output_dir,
                    const ExportConfig& config,
                    const mjModel* model,
                    const mjvCamera& camera,
                    const mjvScene& scene,
                    const mujoco::nelif::GBufferPass& gbuffer,
                    const mujoco::nelif::ScreenSpaceLightPass& screen_space_light) {
  std::error_code error;
  std::filesystem::create_directories(output_dir, error);
  if (error) {
    std::fprintf(stderr, "Failed to create output dir %s: %s\n",
                 output_dir.string().c_str(), error.message().c_str());
    return false;
  }

  std::vector<float> rgba;
  for (int i = 0; i < mujoco::nelif::kGBufferAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::GBufferAttachment>(i);
    if (!gbuffer.Readback(attachment, &rgba) ||
        !DumpAttachment(output_dir, mujoco::nelif::kGBufferContract[i].name,
                        gbuffer.width(), gbuffer.height(), rgba,
                        config.exr_precision)) {
      std::fprintf(stderr, "Failed to export %s\n",
                   mujoco::nelif::kGBufferContract[i].name);
      return false;
    }
  }

  for (int i = 0; i < mujoco::nelif::kScreenSpaceLightAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::ScreenSpaceLightAttachment>(i);
    if (!screen_space_light.Readback(attachment, &rgba) ||
        !DumpAttachment(output_dir, mujoco::nelif::kScreenSpaceLightContract[i].name,
                        screen_space_light.width(), screen_space_light.height(), rgba,
                        config.exr_precision)) {
      std::fprintf(stderr, "Failed to export %s\n",
                   mujoco::nelif::kScreenSpaceLightContract[i].name);
      return false;
    }

    if (config.write_debug &&
        attachment == mujoco::nelif::ScreenSpaceLightAttachment::kSSLightDepth &&
        !DumpSSLightDepthDebug(output_dir, screen_space_light.width(),
                               screen_space_light.height(), rgba)) {
      return false;
    }
  }

  for (int i = 0; i < mujoco::nelif::kLightSpaceAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::LightSpaceAttachment>(i);
    if (!screen_space_light.Readback(attachment, &rgba) ||
        !DumpAttachment(output_dir, mujoco::nelif::kLightSpaceContract[i].name,
                        screen_space_light.light_space_width(),
                        screen_space_light.light_space_height(), rgba,
                        config.exr_precision)) {
      std::fprintf(stderr, "Failed to export %s\n",
                   mujoco::nelif::kLightSpaceContract[i].name);
      return false;
    }
  }

  WriteManifest(output_dir, config, model, camera, scene, gbuffer, screen_space_light);
  return true;
}

bool LoadModel(const std::string& path, mjModel** model, mjData** data) {
  char error[1000] = "Could not load model";
  mjModel* loaded = nullptr;
  if (path.size() > 4 && path.substr(path.size() - 4) == ".mjb") {
    loaded = mj_loadModel(path.c_str(), nullptr);
  } else {
    loaded = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
  }

  if (!loaded) {
    std::fprintf(stderr, "Load model error: %s\n", error);
    return false;
  }

  *model = loaded;
  *data = mj_makeData(loaded);
  mj_forward(*model, *data);
  return true;
}

bool InitCamera(const mjModel* model, mjvCamera* cam,
                const std::optional<std::string>& camera_name) {
  mjv_defaultCamera(cam);

  if (camera_name.has_value()) {
    int camid = mj_name2id(model, mjOBJ_CAMERA, camera_name->c_str());
    if (camid < 0) {
      std::fprintf(stderr, "Could not find camera '%s'\n", camera_name->c_str());
      return false;
    }
    cam->type = mjCAMERA_FIXED;
    cam->fixedcamid = camid;
    return true;
  }

  int camid = mj_name2id(model, mjOBJ_CAMERA, "nelif_fixed_cam");
  if (camid >= 0) {
    cam->type = mjCAMERA_FIXED;
    cam->fixedcamid = camid;
  } else {
    mjv_defaultFreeCamera(model, cam);
  }
  return true;
}

bool ResetToKeyframe(const mjModel* model, mjData* data,
                     const std::optional<std::string>& key_name) {
  if (!key_name.has_value()) {
    return true;
  }

  const int keyid = mj_name2id(model, mjOBJ_KEY, key_name->c_str());
  if (keyid < 0) {
    std::fprintf(stderr, "Could not find keyframe '%s'\n", key_name->c_str());
    return false;
  }

  mj_resetDataKeyframe(model, data, keyid);
  mj_forward(model, data);
  return true;
}

}  // namespace

int main(int argc, const char** argv) {
  ExportConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    PrintUsage();
    return EXIT_FAILURE;
  }
  LogVerbose(config, "parsed arguments");

  nelif_sample::LoadMeshDecoderPlugins(argv[0]);
  LogVerbose(config, "loaded mesh decoder plugins");

  mjModel* model = nullptr;
  mjData* data = nullptr;
  if (!LoadModel(config.model_path, &model, &data)) {
    return EXIT_FAILURE;
  }
  LogVerbose(config, "loaded model");

  if (!ResetToKeyframe(model, data, config.key_name)) {
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }
  LogVerbose(config, "reset model state");

  if (!glfwInit()) {
    std::fprintf(stderr, "Could not initialize GLFW\n");
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }
  LogVerbose(config, "initialized GLFW");

#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif
  glfwWindowHint(GLFW_VISIBLE, config.visible ? GLFW_TRUE : GLFW_FALSE);
  GLFWwindow* window =
      glfwCreateWindow(config.width, config.height, "NeLiF Raw Export", nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "Could not create GLFW window\n");
    glfwTerminate();
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }
  LogVerbose(config, "created GLFW window");

  glfwMakeContextCurrent(window);
  glfwPollEvents();
  LogVerbose(config, "created OpenGL context");

  mjvCamera cam;
  mjvOption opt;
  mjvScene scn;
  mjrContext con;
  mujoco::nelif::GBufferPass gbuffer;
  mujoco::nelif::ScreenSpaceLightPass screen_space_light;

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  if (!InitCamera(model, &cam, config.camera_name)) {
    glfwDestroyWindow(window);
    glfwTerminate();
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }
  LogVerbose(config, "initialized camera");

  mjv_makeScene(model, &scn, 2000);
  mjr_makeContext(model, &con, mjFONTSCALE_150);
  LogVerbose(config, "created MuJoCo render context");

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
  if (framebuffer_width <= 0 || framebuffer_height <= 0) {
    framebuffer_width = config.width;
    framebuffer_height = config.height;
  }

  gbuffer.Init(framebuffer_width, framebuffer_height);
  screen_space_light.Init(framebuffer_width, framebuffer_height, config.face_size);
  LogVerbose(config, "initialized NeLiF passes");

  for (int i = 0; i < config.steps; ++i) {
    mj_step(model, data);
  }
  LogVerbose(config, "advanced simulation");

  mjv_updateScene(model, data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
  LogVerbose(config, "updated scene");
  gbuffer.Render(model, &scn);
  LogVerbose(config, "rendered G-buffer");
  screen_space_light.Render(model, &scn, gbuffer);
  glfwPollEvents();
  LogVerbose(config, "rendered screen/light attachments");

  bool ok = ExportRawFrame(config.output_dir, config, model, cam, scn, gbuffer,
                           screen_space_light);
  LogVerbose(config, ok ? "exported EXRs" : "export failed");

  screen_space_light.Shutdown();
  gbuffer.Shutdown();
  mjr_freeContext(&con);
  mjv_freeScene(&scn);
  glfwDestroyWindow(window);
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif
  mj_deleteData(data);
  mj_deleteModel(model);

  if (!ok) {
    return EXIT_FAILURE;
  }

  std::printf("Exported raw NeLiF EXRs to %s\n", config.output_dir.string().c_str());
  return EXIT_SUCCESS;
}
