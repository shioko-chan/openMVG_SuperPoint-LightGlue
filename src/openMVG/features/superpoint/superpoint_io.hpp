#ifndef OPENMVG_FEATURES_SUPERPOINT_IMAGE_DESCRIBER_IO_HPP
#define OPENMVG_FEATURES_SUPERPOINT_IMAGE_DESCRIBER_IO_HPP

#include "openMVG/features/superpoint/superpoint.hpp"

#include <cereal/types/polymorphic.hpp>

template <class Archive>
void openMVG::features::SuperPoint_Image_describer::serialize(Archive &ar)
{
    ar(cereal::make_nvp("max_h", max_h), cereal::make_nvp("max_w", max_w));
}

CEREAL_REGISTER_TYPE_WITH_NAME(openMVG::features::SuperPoint_Image_describer, "SUPERPOINT_Image_describer");
CEREAL_REGISTER_POLYMORPHIC_RELATION(openMVG::features::Image_describer, openMVG::features::SuperPoint_Image_describer)

#endif
