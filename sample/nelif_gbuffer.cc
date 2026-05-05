// Minimal NeLiF G-buffer validation sample.
//
// This sample intentionally supports only simple primitives first. It is a
// sandbox for validating NeLiF input channels before touching MuJoCo's core
// renderer or adding TensorRT inference.

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include "experimental/nelif/exr_writer.h"
#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/screen_space_light_pass.h"
#include "nelif_decoder_plugins.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

mjModel* m = nullptr;
mjData* d = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

int debug_mode = mujoco::nelif::kDebugPosition;
mujoco::nelif::GBufferPass gbuffer;
mujoco::nelif::ScreenSpaceLightPass screen_space_light;
std::string model_path;
bool dump_requested = false;
int dump_index = 0;
bool dump_first_frame = false;
bool exit_after_dump = false;
bool dump_completed = false;
int window_width = 1200;
int window_height = 900;
int face_size = 512;
std::optional<std::filesystem::path> output_root;

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

std::filesystem::path FindRepoRoot() {
  std::filesystem::path current = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::exists(current / "tmp-res") &&
        std::filesystem::exists(current / "References") &&
        std::filesystem::exists(current / "mujoco")) {
      return current;
    }
    if (current == current.root_path()) {
      return std::filesystem::current_path();
    }
    current = current.parent_path();
  }
}

bool ParseIntFlagValue(const char* value, int* out) {
  if (!value || !out) {
    return false;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed > 1 << 20) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
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

void WriteManifest(const std::filesystem::path& dir) {
  std::ofstream out(dir / "manifest.json");
  if (!out) {
    std::printf("Failed to open manifest for %s\n", dir.string().c_str());
    return;
  }

  out << "{\n";
  out << "  \"model\": \"" << EscapeJson(model_path) << "\",\n";
  out << "  \"screen_width\": " << gbuffer.width() << ",\n";
  out << "  \"screen_height\": " << gbuffer.height() << ",\n";
  out << "  \"light_face_size\": " << screen_space_light.face_size() << ",\n";
  out << "  \"light_space_width\": " << screen_space_light.light_space_width() << ",\n";
  out << "  \"light_space_height\": " << screen_space_light.light_space_height() << ",\n";
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

bool DumpAttachment(const std::filesystem::path& dir, const char* name,
                    int width, int height, const std::vector<float>& rgba) {
  std::string error;
  if (mujoco::nelif::SaveRgba32fExr((dir / (std::string(name) + ".exr")).string(),
                                    width, height, rgba, &error, /*flip_y=*/true)) {
    return true;
  }
  std::printf("%s\n", error.c_str());
  return false;
}

void DumpCurrentFrame() {
  std::filesystem::path repo_root =
      output_root.has_value() ? *output_root : (FindRepoRoot() / "tmp-res" / "nelif_dump");
  char frame_name[32];
  std::snprintf(frame_name, sizeof(frame_name), "frame_%06d", dump_index++);
  std::filesystem::path dump_dir = repo_root / frame_name;
  std::error_code error;
  std::filesystem::create_directories(dump_dir, error);
  if (error) {
    std::printf("Failed to create dump dir %s: %s\n", dump_dir.string().c_str(),
                error.message().c_str());
    return;
  }

  std::vector<float> rgba;
  for (int i = 0; i < mujoco::nelif::kGBufferAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::GBufferAttachment>(i);
    if (!gbuffer.Readback(attachment, &rgba) ||
        !DumpAttachment(dump_dir, mujoco::nelif::kGBufferContract[i].name,
                        gbuffer.width(), gbuffer.height(), rgba)) {
      std::printf("Failed to dump %s\n", mujoco::nelif::kGBufferContract[i].name);
      return;
    }
  }
  for (int i = 0; i < mujoco::nelif::kScreenSpaceLightAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::ScreenSpaceLightAttachment>(i);
    if (!screen_space_light.Readback(attachment, &rgba) ||
        !DumpAttachment(dump_dir, mujoco::nelif::kScreenSpaceLightContract[i].name,
                        screen_space_light.width(), screen_space_light.height(), rgba)) {
      std::printf("Failed to dump %s\n", mujoco::nelif::kScreenSpaceLightContract[i].name);
      return;
    }
  }
  for (int i = 0; i < mujoco::nelif::kLightSpaceAttachmentCount; ++i) {
    auto attachment = static_cast<mujoco::nelif::LightSpaceAttachment>(i);
    if (!screen_space_light.Readback(attachment, &rgba) ||
        !DumpAttachment(dump_dir, mujoco::nelif::kLightSpaceContract[i].name,
                        screen_space_light.light_space_width(),
                        screen_space_light.light_space_height(), rgba)) {
      std::printf("Failed to dump %s\n", mujoco::nelif::kLightSpaceContract[i].name);
      return;
    }
  }

  WriteManifest(dump_dir);
  std::printf("Dumped raw NeLiF attachments to %s\n", dump_dir.string().c_str());
}

void Keyboard(GLFWwindow* window, int key, int, int act, int) {
  if (act != GLFW_PRESS) {
    return;
  }
  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, 1);
  } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
    debug_mode = key - GLFW_KEY_0;
    std::printf("Debug mode %d\n", debug_mode);
  } else if (key == GLFW_KEY_MINUS) {
    debug_mode = mujoco::nelif::kDebugSMNormal;
    std::printf("Debug mode %d\n", debug_mode);
  } else if (key == GLFW_KEY_EQUAL) {
    debug_mode = mujoco::nelif::kDebugSMFlux;
    std::printf("Debug mode %d\n", debug_mode);
  } else if (key == GLFW_KEY_D) {
    dump_requested = true;
    std::printf("Will dump current frame attachments\n");
  } else if (key == GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

void LoadModel(const char* path) {
  char error[1000] = "Could not load model";
  if (std::strlen(path) > 4 && !std::strcmp(path + std::strlen(path) - 4, ".mjb")) {
    m = mj_loadModel(path, nullptr);
  } else {
    m = mj_loadXML(path, nullptr, error, sizeof(error));
  }
  if (!m) {
    mju_error("Load model error: %s", error);
  }
  d = mj_makeData(m);
  mj_forward(m, d);
}

void InitMuJoCoCamera() {
  mjv_defaultCamera(&cam);
  int camid = mj_name2id(m, mjOBJ_CAMERA, "nelif_fixed_cam");
  if (camid >= 0) {
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = camid;
  } else {
    mjv_defaultFreeCamera(m, &cam);
  }
}

}  // namespace

int main(int argc, const char** argv) {
  if (argc < 2) {
    std::printf("USAGE: nelif_gbuffer modelfile [--dump-first-frame] [--exit-after-dump] "
                "[--output-dir dir] [--width px] [--height px] [--face-size px]\n");
    return EXIT_FAILURE;
  }
  model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dump-first-frame")) {
      dump_first_frame = true;
    } else if (!std::strcmp(argv[i], "--exit-after-dump")) {
      exit_after_dump = true;
    } else if (!std::strcmp(argv[i], "--output-dir")) {
      if (i + 1 >= argc) {
        std::printf("Missing value for --output-dir\n");
        return EXIT_FAILURE;
      }
      output_root = std::filesystem::path(argv[++i]);
    } else if (!std::strcmp(argv[i], "--width")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &window_width)) {
        std::printf("Invalid value for --width\n");
        return EXIT_FAILURE;
      }
    } else if (!std::strcmp(argv[i], "--height")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &window_height)) {
        std::printf("Invalid value for --height\n");
        return EXIT_FAILURE;
      }
    } else if (!std::strcmp(argv[i], "--face-size")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &face_size)) {
        std::printf("Invalid value for --face-size\n");
        return EXIT_FAILURE;
      }
    } else {
      std::printf("Unknown flag: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  nelif_sample::LoadMeshDecoderPlugins(argv[0]);
  LoadModel(argv[1]);

  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif
  GLFWwindow* window = glfwCreateWindow(window_width, window_height, "NeLiF G-buffer", nullptr, nullptr);
  if (!window) {
    mju_error("Could not create GLFW window");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetKeyCallback(window, Keyboard);

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
  InitMuJoCoCamera();

  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
  gbuffer.Init(framebuffer_width, framebuffer_height);
  screen_space_light.Init(framebuffer_width, framebuffer_height, face_size);

  std::printf(
      "Debug keys: 0 MuJoCo, 1 Position, 2 Normal, 3 Diffuse, 4 Reflect, 5 Gloss, 6 OutDir, "
      "7 SSLightDepth, 8 SSLightVec, 9 SMPosition, - SMNormal, = SMFlux, D Dump\n");

  while (!glfwWindowShouldClose(window)) {
    mj_step(m, d);

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    gbuffer.Resize(viewport.width, viewport.height);
    gbuffer.Render(m, &scn);
    screen_space_light.Resize(viewport.width, viewport.height);
    screen_space_light.Render(m, &scn, gbuffer);
    if (dump_requested || (dump_first_frame && !dump_completed)) {
      DumpCurrentFrame();
      dump_requested = false;
      dump_completed = true;
      if (exit_after_dump) {
        glfwSetWindowShouldClose(window, 1);
      }
    }

    if (debug_mode == mujoco::nelif::kDebugMujoco) {
      mjr_render(viewport, &scn, &con);
    } else if (debug_mode == mujoco::nelif::kDebugSSLightDepth ||
               debug_mode == mujoco::nelif::kDebugSSLightVec ||
               debug_mode == mujoco::nelif::kDebugSMPosition ||
               debug_mode == mujoco::nelif::kDebugSMNormal ||
               debug_mode == mujoco::nelif::kDebugSMFlux) {
      screen_space_light.DrawDebug(debug_mode, viewport.width, viewport.height);
    } else {
      gbuffer.DrawDebug(debug_mode, viewport.width, viewport.height);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
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

  return EXIT_SUCCESS;
}
