#ifndef MUJOCO_SAMPLE_NELIF_DECODER_PLUGINS_H_
#define MUJOCO_SAMPLE_NELIF_DECODER_PLUGINS_H_

#include <mujoco/mujoco.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace nelif_sample {

inline std::filesystem::path CanonicalOrAbsolute(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
  if (!error) {
    return canonical;
  }

  error.clear();
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (!error) {
    return absolute.lexically_normal();
  }

  return path.lexically_normal();
}

inline void AddPluginDirectory(std::vector<std::filesystem::path>* dirs,
                               const std::filesystem::path& dir) {
  if (!dirs || dir.empty()) {
    return;
  }

  std::error_code error;
  if (!std::filesystem::is_directory(dir, error)) {
    return;
  }

  std::filesystem::path canonical = CanonicalOrAbsolute(dir);
  const std::string key = canonical.string();
  for (const std::filesystem::path& existing : *dirs) {
    if (existing.string() == key) {
      return;
    }
  }
  dirs->push_back(canonical);
}

inline std::filesystem::path ExecutableDirectory(const char* argv0) {
  if (!argv0 || !argv0[0]) {
    return CanonicalOrAbsolute(std::filesystem::current_path());
  }

  std::filesystem::path executable(argv0);
  if (executable.has_parent_path()) {
    return CanonicalOrAbsolute(executable).parent_path();
  }

  return CanonicalOrAbsolute(std::filesystem::current_path());
}

inline bool HasPath(const std::vector<std::filesystem::path>& paths,
                    const std::filesystem::path& candidate) {
  const std::string key = candidate.string();
  for (const std::filesystem::path& path : paths) {
    if (path.string() == key) {
      return true;
    }
  }
  return false;
}

inline void LoadDecoderIfPresent(const std::filesystem::path& library,
                                 std::vector<std::filesystem::path>* loaded) {
  if (!loaded) {
    return;
  }

  std::error_code error;
  if (!std::filesystem::is_regular_file(library, error)) {
    return;
  }

  std::filesystem::path canonical = CanonicalOrAbsolute(library);
  if (HasPath(*loaded, canonical)) {
    return;
  }

  mj_loadPluginLibrary(canonical.string().c_str());
  loaded->push_back(canonical);
}

inline void LoadMeshDecoderPlugins(const char* argv0) {
  std::vector<std::filesystem::path> plugin_dirs;

  if (const char* env_dir = std::getenv("MUJOCO_PLUGIN_DIR")) {
    AddPluginDirectory(&plugin_dirs, env_dir);
  }

  const std::filesystem::path executable_dir = ExecutableDirectory(argv0);
  AddPluginDirectory(&plugin_dirs, executable_dir / "../lib");
  AddPluginDirectory(&plugin_dirs, executable_dir / "../mujoco_plugin");
  AddPluginDirectory(&plugin_dirs, executable_dir / "mujoco_plugin");

  const std::filesystem::path cwd = CanonicalOrAbsolute(std::filesystem::current_path());
  AddPluginDirectory(&plugin_dirs, cwd / "mujoco/build/lib");
  AddPluginDirectory(&plugin_dirs, cwd / "build/lib");

#if defined(_WIN32) || defined(__CYGWIN__)
  const char* decoder_libraries[] = {
      "obj_decoder.dll", "stl_decoder.dll",
      "libobj_decoder.dll", "libstl_decoder.dll",
  };
#elif defined(__APPLE__)
  const char* decoder_libraries[] = {
      "libobj_decoder.dylib", "libstl_decoder.dylib",
  };
#else
  const char* decoder_libraries[] = {
      "libobj_decoder.so", "libstl_decoder.so",
  };
#endif

  std::vector<std::filesystem::path> loaded_libraries;
  for (const std::filesystem::path& dir : plugin_dirs) {
    for (const char* decoder_library : decoder_libraries) {
      LoadDecoderIfPresent(dir / decoder_library, &loaded_libraries);
    }
  }
}

}  // namespace nelif_sample

#endif  // MUJOCO_SAMPLE_NELIF_DECODER_PLUGINS_H_
