from tracking_engine.reid.backbones.osnet import (
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
)
from tracking_engine.reid.backbones.osnet_ain import (
    osnet_ain_x0_25,
    osnet_ain_x0_5,
    osnet_ain_x0_75,
    osnet_ain_x1_0,
)

# Dùng các biến thể OSNet thường cho ReID.
MODEL_FACTORY = {
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ain_x1_0": osnet_ain_x1_0,
    "osnet_ain_x0_75": osnet_ain_x0_75,
    "osnet_ain_x0_5": osnet_ain_x0_5,
    "osnet_ain_x0_25": osnet_ain_x0_25,
}

