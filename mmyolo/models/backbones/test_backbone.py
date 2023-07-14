from mmyolo.models import YOLOXCSPDarknet,YOLOv6EfficientRep
import torch
deepen_factor = 0.33
widen_factor = 0.375
model = YOLOXCSPDarknet(out_indices=[1,2,3,4],
                        widen_factor=widen_factor,
                        deepen_factor=deepen_factor)
# model = YOLOv6EfficientRep(out_indices=[1,2,3,4])
model.eval()
inputs = torch.rand(1, 3, 416, 416)
level_outputs = model(inputs)
for level_out in level_outputs:
     print(tuple(level_out.shape))