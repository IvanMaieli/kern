//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#include <cstddef>
#include <string_view>

namespace kern {
    enum class DataType : std::uint8_t {
        float32,
        float16,
        int8,
    };

    [[nodiscard]] constexpr std::size_t size_bytes(const DataType data_type) {
        switch (data_type) {
            case DataType::float32:
                return 4;
            case DataType::float16:
                return 2;
            case DataType::int8:
                return 1;
        }
        return 0;
    }

    [[nodiscard]] constexpr std::string_view name(const DataType data_type) {
        switch (data_type) {
            case DataType::float32:
                return "float32";
            case DataType::float16:
                return "float16";
            case DataType::int8:
                return "int8";
        }
        return "unknown";
    }

} // namespace kern