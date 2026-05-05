#ifndef MUJOCO_EXPERIMENTAL_NELIF_EXR_WRITER_H_
#define MUJOCO_EXPERIMENTAL_NELIF_EXR_WRITER_H_

#include <string>
#include <vector>

namespace mujoco::nelif {

bool SaveRgba32fExr(const std::string& path, int width, int height,
                    const float* rgba, std::string* error, bool flip_y = false,
                    bool save_as_fp16 = false);

bool SaveRgba32fExr(const std::string& path, int width, int height,
                    const std::vector<float>& rgba, std::string* error,
                    bool flip_y = false, bool save_as_fp16 = false);

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_EXR_WRITER_H_
