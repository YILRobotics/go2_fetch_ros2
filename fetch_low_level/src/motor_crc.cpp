#include "fetch_low_level/motor_crc.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace fetch_low_level {
namespace {
// Explicit wire-layout mirrors the Unitree DDS LowCmd representation. ROS
// message classes are not guaranteed to have a contiguous wire-compatible
// memory layout, so CRC must be calculated from this packed-by-construction copy.
struct MotorCmdRaw
{
  uint8_t mode;
  uint8_t pad[3];
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  uint32_t reserve[3];
};

struct BmsCmdRaw
{
  uint8_t off;
  uint8_t reserve[3];
};

struct LowCmdRaw
{
  uint8_t head[2];
  uint8_t level_flag;
  uint8_t frame_reserve;
  uint32_t sn[2];
  uint32_t version[2];
  uint16_t bandwidth;
  uint8_t pad[2];
  MotorCmdRaw motor_cmd[20];
  BmsCmdRaw bms_cmd;
  uint8_t wireless_remote[40];
  uint8_t led[12];
  uint8_t fan[2];
  uint8_t gpio;
  uint8_t pad2;
  uint32_t reserve;
  uint32_t crc;
};

// Detect compiler padding changes that would silently produce invalid CRCs.
static_assert(sizeof(LowCmdRaw) == 812, "Unexpected Unitree LowCmd wire layout");

uint32_t crc32_core(const uint32_t *ptr, uint32_t len)
{
  uint32_t crc = 0xFFFFFFFFU;
  constexpr uint32_t polynomial = 0x04C11DB7U;
  for (uint32_t i = 0; i < len; ++i)
  {
    uint32_t data = ptr[i];
    for (uint32_t bit = 0x80000000U; bit != 0; bit >>= 1U)
    {
      const bool top = (crc & 0x80000000U) != 0;
      crc <<= 1U;
      if (top)
        crc ^= polynomial;
      if ((data & bit) != 0)
        crc ^= polynomial;
    }
  }
  return crc;
}
}

void set_crc(unitree_go::msg::LowCmd &msg)
{
  // CRC covers every 32-bit word except the final CRC field itself.
  LowCmdRaw raw{};
  std::copy(msg.head.begin(), msg.head.end(), raw.head);
  raw.level_flag = msg.level_flag;
  raw.frame_reserve = msg.frame_reserve;
  std::copy(msg.sn.begin(), msg.sn.end(), raw.sn);
  std::copy(msg.version.begin(), msg.version.end(), raw.version);
  raw.bandwidth = msg.bandwidth;
  for (size_t i = 0; i < 20; ++i)
  {
    const auto &s = msg.motor_cmd[i];
    auto &d = raw.motor_cmd[i];
    d.mode = s.mode;
    d.q = s.q;
    d.dq = s.dq;
    d.tau = s.tau;
    d.kp = s.kp;
    d.kd = s.kd;
    std::copy(s.reserve.begin(), s.reserve.end(), d.reserve);
  }
  raw.bms_cmd.off = msg.bms_cmd.off;
  std::copy(msg.bms_cmd.reserve.begin(), msg.bms_cmd.reserve.end(), raw.bms_cmd.reserve);
  std::copy(msg.wireless_remote.begin(), msg.wireless_remote.end(), raw.wireless_remote);
  std::copy(msg.led.begin(), msg.led.end(), raw.led);
  std::copy(msg.fan.begin(), msg.fan.end(), raw.fan);
  raw.gpio = msg.gpio;
  raw.reserve = msg.reserve;
  msg.crc = crc32_core(reinterpret_cast<const uint32_t *>(&raw), sizeof(raw) / 4U - 1U);
}
}
