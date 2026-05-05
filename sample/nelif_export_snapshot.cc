// One-shot scene snapshot exporter for downstream Cycles/Blender GT generation.
//
// This tool does not render. It loads a MuJoCo model, advances the simulation
// if requested, updates the visualization scene, and writes a compact JSON
// snapshot that contains the active camera, the first active local light, and
// the supported world-space geoms used by the current NeLiF raw exporters.
// Mesh geoms are baked into vertices/faces/normals from the compiled mjModel
// so the downstream Blender scene uses the same geometry as MuJoCo.

#include <mujoco/mujoco.h>

#include "nelif_decoder_plugins.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace {

enum class MeshMode {
  kEmbedded = 0,
  kCache,
};

struct SnapshotConfig {
  std::string model_path;
  std::filesystem::path output_path;
  std::filesystem::path mesh_cache_dir;
  std::optional<std::string> camera_name;
  std::optional<std::string> key_name;
  int steps = 0;
  MeshMode mesh_mode = MeshMode::kEmbedded;
};

struct Vec3 {
  float x;
  float y;
  float z;
};

Vec3 Cross(Vec3 a, Vec3 b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

float Dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Normalize(Vec3 v) {
  const float len = std::sqrt(Dot(v, v));
  if (len <= 1e-8f) {
    return {0.0f, 0.0f, 1.0f};
  }
  return {v.x / len, v.y / len, v.z / len};
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
      "USAGE: nelif_export_snapshot modelfile --output path "
      "[--steps n] [--camera name] [--key name] "
      "[--mesh-mode embedded|cache] [--mesh-cache-dir dir]\n");
}

bool ParseMeshMode(const char* value, MeshMode* out) {
  if (!value || !out) {
    return false;
  }
  if (!std::strcmp(value, "embedded")) {
    *out = MeshMode::kEmbedded;
    return true;
  }
  if (!std::strcmp(value, "cache")) {
    *out = MeshMode::kCache;
    return true;
  }
  return false;
}

const char* MeshModeName(MeshMode mode) {
  switch (mode) {
    case MeshMode::kEmbedded:
      return "embedded";
    case MeshMode::kCache:
      return "cache";
  }
  return "embedded";
}

bool ParseArgs(int argc, const char** argv, SnapshotConfig* config) {
  if (!config || argc < 2) {
    return false;
  }

  config->model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--output")) {
      if (i + 1 >= argc) {
        return false;
      }
      config->output_path = std::filesystem::path(argv[++i]);
    } else if (!std::strcmp(argv[i], "--steps")) {
      if (i + 1 >= argc || !ParseIntFlagValue(argv[++i], &config->steps)) {
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
    } else if (!std::strcmp(argv[i], "--mesh-mode")) {
      if (i + 1 >= argc || !ParseMeshMode(argv[++i], &config->mesh_mode)) {
        return false;
      }
    } else if (!std::strcmp(argv[i], "--mesh-cache-dir")) {
      if (i + 1 >= argc) {
        return false;
      }
      config->mesh_cache_dir = std::filesystem::path(argv[++i]);
    } else {
      return false;
    }
  }

  return !config->output_path.empty();
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

const char* SafeName(const char* name) {
  return name ? name : "";
}

const char* GeomTypeName(int type) {
  switch (type) {
    case mjGEOM_PLANE:
      return "plane";
    case mjGEOM_BOX:
      return "box";
    case mjGEOM_SPHERE:
      return "sphere";
    case mjGEOM_MESH:
      return "mesh";
    default:
      return "unsupported";
  }
}

constexpr float kFiniteRadiusEps = 1e-6f;

const char* LightTypeName(const mjvLight& light) {
  switch (light.type) {
    case mjLIGHT_SPOT:
      return "spot";
    case mjLIGHT_DIRECTIONAL:
      return "directional";
    default:
      if (light.bulbradius > kFiniteRadiusEps) {
        return "sphere_area";
      }
      return "point";
  }
}

float ComputeRoughness(const mjvGeom& geom) {
  float roughness = std::sqrt(1.0f / std::fmax(geom.shininess * 128.0f, 1.0f));
  return std::fmin(std::fmax(roughness, 0.02f), 1.0f);
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

bool IsSupportedGeom(const mjvGeom& geom) {
  if (geom.category == mjCAT_DECOR || geom.rgba[3] <= 0.0f) {
    return false;
  }
  return geom.type == mjGEOM_PLANE || geom.type == mjGEOM_BOX ||
         geom.type == mjGEOM_SPHERE || geom.type == mjGEOM_MESH;
}

int MeshIdFromGeom(const mjvGeom& geom) {
  if (geom.type != mjGEOM_MESH || geom.dataid < 0) {
    return -1;
  }
  return geom.dataid / 2;
}

std::string SanitizePathComponent(const std::string& input) {
  std::string sanitized;
  sanitized.reserve(input.size());
  for (char c : input) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
      sanitized.push_back(c);
    } else {
      sanitized.push_back('_');
    }
  }
  return sanitized.empty() ? "unnamed" : sanitized;
}

std::filesystem::path MeshCacheDir(const SnapshotConfig& config) {
  if (!config.mesh_cache_dir.empty()) {
    return config.mesh_cache_dir;
  }
  return config.output_path.parent_path() / "compiled_meshes";
}

uint64_t HashBytes(uint64_t hash, const void* data, size_t size) {
  const unsigned char* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t MeshHash(const mjModel* model, int mesh_id) {
  uint64_t hash = 1469598103934665603ull;
  const int vert_adr = model->mesh_vertadr[mesh_id];
  const int vert_num = model->mesh_vertnum[mesh_id];
  const int face_adr = model->mesh_faceadr[mesh_id];
  const int face_num = model->mesh_facenum[mesh_id];
  hash = HashBytes(hash, model->mesh_vert + 3 * vert_adr,
                   static_cast<size_t>(vert_num) * 3 * sizeof(float));
  hash = HashBytes(hash, model->mesh_face + 3 * face_adr,
                   static_cast<size_t>(face_num) * 3 * sizeof(int));
  const int normal_adr = model->mesh_normaladr[mesh_id];
  if (normal_adr >= 0 && model->mesh_normalnum[mesh_id] > 0) {
    hash = HashBytes(hash, model->mesh_normal + 3 * normal_adr,
                     static_cast<size_t>(model->mesh_normalnum[mesh_id]) * 3 * sizeof(float));
  }
  if (model->mesh_facenormal) {
    hash = HashBytes(hash, model->mesh_facenormal + 3 * face_adr,
                     static_cast<size_t>(face_num) * 3 * sizeof(int));
  }
  return hash;
}

std::filesystem::path MeshCachePath(const SnapshotConfig& config, const mjModel* model,
                                    const char* mesh_name, int mesh_id) {
  std::ostringstream filename;
  filename << "mesh_" << mesh_id << "_" << SanitizePathComponent(SafeName(mesh_name))
           << "_" << std::hex << MeshHash(model, mesh_id) << ".mesh.json";
  return MeshCacheDir(config) / filename.str();
}

std::string SnapshotRelativePath(const std::filesystem::path& snapshot_path,
                                 const std::filesystem::path& target_path) {
  std::error_code error;
  const std::filesystem::path relative =
      std::filesystem::relative(target_path, snapshot_path.parent_path(), error);
  if (!error) {
    return relative.generic_string();
  }
  return target_path.generic_string();
}

const char* MeshSourcePath(const mjModel* model, int mesh_id) {
  if (!model || mesh_id < 0 || mesh_id >= model->nmesh ||
      !model->mesh_pathadr || !model->paths || model->mesh_pathadr[mesh_id] < 0) {
    return nullptr;
  }
  return model->paths + model->mesh_pathadr[mesh_id];
}

void WriteMeshObject(std::ofstream& out, const mjModel* model, int mesh_id,
                     const std::string& indent) {
  const int vert_adr = model->mesh_vertadr[mesh_id];
  const int vert_num = model->mesh_vertnum[mesh_id];
  const int face_adr = model->mesh_faceadr[mesh_id];
  const int face_num = model->mesh_facenum[mesh_id];
  const int normal_adr = model->mesh_normaladr[mesh_id];
  const bool has_normals =
      normal_adr >= 0 && model->mesh_normalnum[mesh_id] > 0 &&
      model->mesh_facenormal != nullptr;

  out << "{\n";
  out << indent << "  \"vertices\": [\n";
  for (int i = 0; i < vert_num; ++i) {
    const float* v = model->mesh_vert + 3 * (vert_adr + i);
    out << indent << "    [" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    if (i + 1 < vert_num) {
      out << ",";
    }
    out << "\n";
  }
  out << indent << "  ],\n";

  out << indent << "  \"faces\": [\n";
  for (int i = 0; i < face_num; ++i) {
    const int face = 3 * (face_adr + i);
    out << indent << "    [" << model->mesh_face[face + 0] << ", "
        << model->mesh_face[face + 1] << ", "
        << model->mesh_face[face + 2] << "]";
    if (i + 1 < face_num) {
      out << ",";
    }
    out << "\n";
  }
  out << indent << "  ],\n";

  out << indent << "  \"corner_normals\": [\n";
  for (int i = 0; i < face_num; ++i) {
    const int face = 3 * (face_adr + i);
    const int i0 = model->mesh_face[face + 0];
    const int i1 = model->mesh_face[face + 1];
    const int i2 = model->mesh_face[face + 2];
    const float* v0 = model->mesh_vert + 3 * (vert_adr + i0);
    const float* v1 = model->mesh_vert + 3 * (vert_adr + i1);
    const float* v2 = model->mesh_vert + 3 * (vert_adr + i2);
    const Vec3 edge0 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const Vec3 edge1 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const Vec3 face_normal = Normalize(Cross(edge0, edge1));

    out << indent << "    [";
    for (int corner = 0; corner < 3; ++corner) {
      Vec3 normal = face_normal;
      if (has_normals) {
        const int normal_index = model->mesh_facenormal[face + corner];
        const float* n = model->mesh_normal + 3 * (normal_adr + normal_index);
        Vec3 mesh_normal = Normalize({n[0], n[1], n[2]});
        if (Dot(face_normal, mesh_normal) >= 0.8f) {
          normal = mesh_normal;
        }
      }
      out << "[" << normal.x << ", " << normal.y << ", " << normal.z << "]";
      if (corner + 1 < 3) {
        out << ", ";
      }
    }
    out << "]";
    if (i + 1 < face_num) {
      out << ",";
    }
    out << "\n";
  }
  out << indent << "  ]\n";
  out << indent << "}";
}

bool WriteMeshCacheFile(const SnapshotConfig& config, const mjModel* model,
                        int mesh_id, const char* mesh_name) {
  const std::filesystem::path cache_path = MeshCachePath(config, model, mesh_name, mesh_id);
  std::error_code error;
  if (std::filesystem::exists(cache_path, error) && !error) {
    return true;
  }
  std::filesystem::create_directories(cache_path.parent_path(), error);
  if (error) {
    std::fprintf(stderr, "Failed to create mesh cache dir %s: %s\n",
                 cache_path.parent_path().string().c_str(), error.message().c_str());
    return false;
  }

  std::ofstream mesh_out(cache_path);
  if (!mesh_out) {
    std::fprintf(stderr, "Failed to open mesh cache %s for writing\n",
                 cache_path.string().c_str());
    return false;
  }

  mesh_out << "{\n";
  mesh_out << "  \"schema\": \"nelif.compiled_mesh.v1\",\n";
  mesh_out << "  \"mesh_id\": " << mesh_id << ",\n";
  mesh_out << "  \"mesh_name\": \"" << EscapeJson(SafeName(mesh_name)) << "\",\n";
  const char* source_path = MeshSourcePath(model, mesh_id);
  mesh_out << "  \"source_path\": "
           << (source_path ? ("\"" + EscapeJson(source_path) + "\"") : "null")
           << ",\n";
  mesh_out << "  \"mesh\": ";
  WriteMeshObject(mesh_out, model, mesh_id, "  ");
  mesh_out << "\n";
  mesh_out << "}\n";
  return true;
}

bool WriteMeshField(std::ofstream& out, const SnapshotConfig& config,
                    const mjModel* model, int mesh_id, const char* mesh_name) {
  if (config.mesh_mode == MeshMode::kEmbedded) {
    out << "      \"mesh_payload\": \"embedded\",\n";
    out << "      \"mesh\": ";
    WriteMeshObject(out, model, mesh_id, "      ");
    out << ",\n";
    return true;
  }

  if (!WriteMeshCacheFile(config, model, mesh_id, mesh_name)) {
    return false;
  }
  const std::filesystem::path cache_path = MeshCachePath(config, model, mesh_name, mesh_id);
  out << "      \"mesh_payload\": \"cache\",\n";
  out << "      \"mesh_ref\": \""
      << EscapeJson(SnapshotRelativePath(config.output_path, cache_path)) << "\",\n";
  return true;
}

bool WriteSnapshot(const std::filesystem::path& output_path,
                   const SnapshotConfig& config,
                   const mjModel* model,
                   const mjvScene& scene) {
  std::error_code error;
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path(), error);
  }
  if (error) {
    std::fprintf(stderr, "Failed to create snapshot dir %s: %s\n",
                 output_path.parent_path().string().c_str(), error.message().c_str());
    return false;
  }

  std::ofstream out(output_path);
  if (!out) {
    std::fprintf(stderr, "Failed to open %s for writing\n", output_path.string().c_str());
    return false;
  }

  mjvGLCamera camera = mjv_averageCamera(scene.camera, scene.camera + 1);

  const mjvLight* active_light = nullptr;
  for (int i = 0; i < scene.nlight; ++i) {
    const mjvLight& candidate = scene.lights[i];
    if (candidate.headlight || candidate.type == mjLIGHT_DIRECTIONAL) {
      continue;
    }
    active_light = &candidate;
    break;
  }

  out << "{\n";
  out << "  \"schema\": \"nelif.snapshot.v1\",\n";
  out << "  \"source\": {\n";
  out << "    \"model\": \"" << EscapeJson(config.model_path) << "\",\n";
  out << "    \"steps\": " << config.steps << ",\n";
  out << "    \"mesh_mode\": \"" << MeshModeName(config.mesh_mode) << "\",\n";
  out << "    \"mesh_cache_dir\": ";
  if (config.mesh_mode == MeshMode::kCache) {
    out << "\"" << EscapeJson(MeshCacheDir(config).generic_string()) << "\",\n";
  } else {
    out << "null,\n";
  }
  out << "    \"camera_name\": "
      << (config.camera_name.has_value()
              ? ("\"" + EscapeJson(*config.camera_name) + "\"")
              : "null")
      << ",\n";
  out << "    \"keyframe\": "
      << (config.key_name.has_value() ? ("\"" + EscapeJson(*config.key_name) + "\"") : "null")
      << "\n";
  out << "  },\n";
  out << "  \"units\": {\n";
  out << "    \"distance\": \"mujoco_world_unit\",\n";
  out << "    \"angles\": \"degrees\",\n";
  out << "    \"color_space\": \"linear\"\n";
  out << "  },\n";
  out << "  \"camera\": {\n";
  out << "    \"position\": [" << camera.pos[0] << ", " << camera.pos[1] << ", "
      << camera.pos[2] << "],\n";
  out << "    \"forward\": [" << camera.forward[0] << ", " << camera.forward[1] << ", "
      << camera.forward[2] << "],\n";
  out << "    \"up\": [" << camera.up[0] << ", " << camera.up[1] << ", " << camera.up[2]
      << "],\n";
  out << "    \"orthographic\": " << (camera.orthographic ? "true" : "false") << ",\n";
  out << "    \"vertical_fov_degrees\": " << ComputeVerticalFovDegrees(camera) << ",\n";
  out << "    \"frustum\": {\n";
  out << "      \"center\": " << camera.frustum_center << ",\n";
  out << "      \"width\": " << camera.frustum_width << ",\n";
  out << "      \"bottom\": " << camera.frustum_bottom << ",\n";
  out << "      \"top\": " << camera.frustum_top << ",\n";
  out << "      \"near\": " << camera.frustum_near << ",\n";
  out << "      \"far\": " << camera.frustum_far << "\n";
  out << "    }\n";
  out << "  },\n";
  out << "  \"active_light\": ";
  if (!active_light) {
    out << "null,\n";
  } else {
    out << "{\n";
    out << "    \"type\": \"" << LightTypeName(*active_light) << "\",\n";
    out << "    \"position\": [" << active_light->pos[0] << ", " << active_light->pos[1]
        << ", " << active_light->pos[2] << "],\n";
    out << "    \"direction\": [" << active_light->dir[0] << ", " << active_light->dir[1]
        << ", " << active_light->dir[2] << "],\n";
    out << "    \"color\": [" << active_light->diffuse[0] << ", "
        << active_light->diffuse[1] << ", " << active_light->diffuse[2] << "],\n";
    const float brightness =
        active_light->intensity > 0.0f ? active_light->intensity : 1.0f;
    out << "    \"intensity\": " << brightness << ",\n";
    out << "    \"brightness\": " << brightness << ",\n";
    out << "    \"emission_rgb\": [" << active_light->diffuse[0] * brightness << ", "
        << active_light->diffuse[1] * brightness << ", "
        << active_light->diffuse[2] * brightness << "],\n";
    out << "    \"radius\": " << active_light->bulbradius << ",\n";
    out << "    \"range\": " << active_light->range << ",\n";
    out << "    \"cone_angle_degrees\": " << active_light->cutoff << "\n";
    out << "  },\n";
  }
  out << "  \"geometries\": [\n";

  bool first = true;
  for (int i = 0; i < scene.ngeom; ++i) {
    const mjvGeom& geom = scene.geoms[i];
    if (!IsSupportedGeom(geom)) {
      continue;
    }
    const int mesh_id = MeshIdFromGeom(geom);
    if (geom.type == mjGEOM_MESH && (mesh_id < 0 || mesh_id >= model->nmesh)) {
      continue;
    }

    const char* geom_name =
        geom.objtype == mjOBJ_UNKNOWN ? nullptr : mj_id2name(model, geom.objtype, geom.objid);
    const char* material_name =
        geom.matid >= 0 ? mj_id2name(model, mjOBJ_MATERIAL, geom.matid) : nullptr;
    const char* mesh_name =
        mesh_id >= 0 ? mj_id2name(model, mjOBJ_MESH, mesh_id) : nullptr;

    if (!first) {
      out << ",\n";
    }
    first = false;

    out << "    {\n";
    out << "      \"name\": \"" << EscapeJson(SafeName(geom_name)) << "\",\n";
    out << "      \"type\": \"" << GeomTypeName(geom.type) << "\",\n";
    out << "      \"objtype\": " << geom.objtype << ",\n";
    out << "      \"objid\": " << geom.objid << ",\n";
    out << "      \"dataid\": " << geom.dataid << ",\n";
    out << "      \"mesh_id\": "
        << (mesh_id >= 0 ? std::to_string(mesh_id) : "null") << ",\n";
    out << "      \"mesh_name\": "
        << (mesh_name ? ("\"" + EscapeJson(mesh_name) + "\"") : "null") << ",\n";
    out << "      \"material_name\": \"" << EscapeJson(SafeName(material_name)) << "\",\n";
    out << "      \"position\": [" << geom.pos[0] << ", " << geom.pos[1] << ", "
        << geom.pos[2] << "],\n";
    out << "      \"size\": [" << geom.size[0] << ", " << geom.size[1] << ", "
        << geom.size[2] << "],\n";
    out << "      \"rotation_matrix\": [\n";
    out << "        [" << geom.mat[0] << ", " << geom.mat[1] << ", " << geom.mat[2] << "],\n";
    out << "        [" << geom.mat[3] << ", " << geom.mat[4] << ", " << geom.mat[5] << "],\n";
    out << "        [" << geom.mat[6] << ", " << geom.mat[7] << ", " << geom.mat[8] << "]\n";
    out << "      ],\n";
    if (geom.type == mjGEOM_MESH) {
      if (!WriteMeshField(out, config, model, mesh_id, mesh_name)) {
        return false;
      }
    }
    out << "      \"material\": {\n";
    out << "        \"rgba\": [" << geom.rgba[0] << ", " << geom.rgba[1] << ", "
        << geom.rgba[2] << ", " << geom.rgba[3] << "],\n";
    out << "        \"diffuse\": [" << geom.rgba[0] << ", " << geom.rgba[1] << ", "
        << geom.rgba[2] << "],\n";
    out << "        \"alpha\": " << geom.rgba[3] << ",\n";
    out << "        \"specular\": " << geom.specular << ",\n";
    out << "        \"shininess\": " << geom.shininess << ",\n";
    out << "        \"roughness\": " << ComputeRoughness(geom) << ",\n";
    out << "        \"reflectance\": " << geom.reflectance << ",\n";
    out << "        \"emission\": " << geom.emission << "\n";
    out << "      }\n";
    out << "    }";
  }

  out << "\n  ]\n";
  out << "}\n";
  return true;
}

}  // namespace

int main(int argc, const char** argv) {
  SnapshotConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    PrintUsage();
    return EXIT_FAILURE;
  }

  nelif_sample::LoadMeshDecoderPlugins(argv[0]);

  mjModel* model = nullptr;
  mjData* data = nullptr;
  if (!LoadModel(config.model_path, &model, &data)) {
    return EXIT_FAILURE;
  }

  if (!ResetToKeyframe(model, data, config.key_name)) {
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }

  mjvCamera cam;
  mjvOption opt;
  mjvScene scene;
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scene);

  if (!InitCamera(model, &cam, config.camera_name)) {
    mj_deleteData(data);
    mj_deleteModel(model);
    return EXIT_FAILURE;
  }

  mjv_makeScene(model, &scene, 2000);

  for (int i = 0; i < config.steps; ++i) {
    mj_step(model, data);
  }

  mjv_updateScene(model, data, &opt, nullptr, &cam, mjCAT_ALL, &scene);
  bool ok = WriteSnapshot(config.output_path, config, model, scene);

  mjv_freeScene(&scene);
  mj_deleteData(data);
  mj_deleteModel(model);

  if (!ok) {
    return EXIT_FAILURE;
  }

  std::printf("Exported NeLiF scene snapshot to %s\n", config.output_path.string().c_str());
  return EXIT_SUCCESS;
}
