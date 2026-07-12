#pragma once

#include <cstddef>
#include <unitree_go/msg/low_cmd.hpp>

namespace fetch_policy {
void set_crc(unitree_go::msg::LowCmd &msg);
}
