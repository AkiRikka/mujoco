#include "experimental/nelif/exr_writer.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include <zlib.h>

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_OPENMP 0
#define TINYEXR_IMPLEMENTATION
#include "experimental/nelif/third_party/tinyexr/tinyexr.h"

namespace mujoco::nelif {

namespace {

std::string FormatTinyExrError(const char* tinyexr_error) {
  if (!tinyexr_error || !tinyexr_error[0]) {
    return "TinyEXR returned an unknown error";
  }
  return std::string(tinyexr_error);
}

std::vector<float> FlipRowsRgba32f(const float* rgba, int width, int height) {
  const size_t row_size = static_cast<size_t>(width) * 4;
  std::vector<float> flipped(static_cast<size_t>(height) * row_size, 0.0f);
  for (int y = 0; y < height; ++y) {
    const size_t src_row = static_cast<size_t>(height - 1 - y) * row_size;
    const size_t dst_row = static_cast<size_t>(y) * row_size;
    std::copy(rgba + src_row, rgba + src_row + row_size, flipped.begin() + dst_row);
  }
  return flipped;
}

}  // namespace

bool SaveRgba32fExr(const std::string& path, int width, int height,
                    const float* rgba, std::string* error, bool flip_y,
                    bool save_as_fp16) {
  if (!rgba || width <= 0 || height <= 0) {
    if (error) {
      *error = "invalid image dimensions or null RGBA buffer";
    }
    return false;
  }

  std::vector<float> flipped_storage;
  const float* source = rgba;
  if (flip_y) {
    flipped_storage = FlipRowsRgba32f(rgba, width, height);
    source = flipped_storage.data();
  }

  const char* tinyexr_error = nullptr;
  int result = SaveEXR(source, width, height, /*components=*/4,
                       save_as_fp16 ? 1 : 0, path.c_str(), &tinyexr_error);
  if (result == TINYEXR_SUCCESS) {
    if (error) {
      error->clear();
    }
    return true;
  }

  if (error) {
    std::ostringstream oss;
    oss << "failed to save EXR '" << path << "': "
        << FormatTinyExrError(tinyexr_error);
    *error = oss.str();
  }
  if (tinyexr_error) {
    FreeEXRErrorMessage(tinyexr_error);
  }
  return false;
}

bool SaveRgba32fExr(const std::string& path, int width, int height,
                    const std::vector<float>& rgba, std::string* error,
                    bool flip_y, bool save_as_fp16) {
  const size_t expected =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  if (rgba.size() != expected) {
    if (error) {
      std::ostringstream oss;
      oss << "RGBA buffer size mismatch: got " << rgba.size()
          << ", expected " << expected;
      *error = oss.str();
    }
    return false;
  }

  return SaveRgba32fExr(path, width, height, rgba.data(), error, flip_y,
                        save_as_fp16);
}

}  // namespace mujoco::nelif
