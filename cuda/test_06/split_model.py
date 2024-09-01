import onnx

# 将原始onnx模型截断，假设有一个节点不能支持，那么需要将网络从该节点之前，之后分别进行子网络提取，然后使用其他方式推理不支持的算子

input_path = 'FACE_YOLOV5s.onnx'
output_path_head = 'FACE_YOLOV5s_HEAD.onnx'
input_names_head = ['images', ]
output_names_head = ['input.64', ]


output_path_tail = 'FACE_YOLOV5s_TAIL.onnx'
input_names_tail = ['input.68', ]
output_names_tail = ['output0', ]

output_path_unsupport = 'FACE_YOLOV5s_UNSUPPORT.onnx'
input_names_unsupport = ['input.64', ]
output_names_unsupport = ['input.68', ]

onnx.utils.extract_model(input_path, output_path_head, input_names_head, output_names_head)
onnx.utils.extract_model(input_path, output_path_tail, input_names_tail, output_names_tail)
onnx.utils.extract_model(input_path, output_path_unsupport, input_names_unsupport, output_names_unsupport)
