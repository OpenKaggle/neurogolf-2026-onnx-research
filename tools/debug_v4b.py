import sys, pathlib
import numpy as np
import onnxruntime

sys.path.insert(0, str(pathlib.Path('tools').resolve()))
from neurogolf_local import encode_grid, all_examples

# Build model with intermediates exposed
import onnx
from onnx import helper as oh, TensorProto

exec(open('tools/build_task233.py').read().split("def main")[0] + '''
def build_debug(out_path):
    m = build(out_path)
    # Add debug outputs
    vi_inside = oh.make_tensor_value_info("inside_bbox", TensorProto.BOOL, [1, 1, 30, 30])
    vi_main_mask = oh.make_tensor_value_info("main_mask", TensorProto.FLOAT, [1, 1, 30, 30])
    vi_labels = oh.make_tensor_value_info("labels_final", TensorProto.FLOAT, [1, 1, 30, 30])
    m.graph.output.append(vi_inside)
    m.graph.output.append(vi_main_mask)
    # Can't easily add labels since it's an iter var - just add inside and main_mask
    onnx.checker.check_model(m, full_check=True)
    return m
''')

# But actually let's just modify the builder to output more info
# Instead, let me just manually check some things

import onnxruntime as ort
from onnx import numpy_helper as onh

# Re-run, get the model, check scalar outputs
models = onnx.load("submissions/handbuilds/task233.onnx")

# Find all initializers to understand values
for init in models.graph.initializer:
    if init.name == 'NEG_BIG':
        neg_big = onh.to_array(init)
        print(f'NEG_BIG = {neg_big}')

# Let me check labels by creating a simpler modified graph
# Actually let me just add debug outputs to the model

# Read the original builder
import tools.build_task233 as builder
# Rebuild and modify
m = builder.build(pathlib.Path("submissions/handbuilds/task233_debug.onnx"))
# Add intermediate outputs
# labels_final is "nlabels_30" which is "nlabels_30" in the graph
# After the loop, nlabels_30 should have the final labels
nodes_list = list(m.graph.node)
# Find where labels_final is computed
last_nlabels = None
for i, node in enumerate(nodes_list):
    if node.name.startswith('nlabels_'):
        last_nlabels = node.name
        break
print(f"Last nlabels node: {last_nlabels}")

# Let me just check by slicing the graph
# Actually the simplest approach: add debug outputs to the original build file
# and rebuild. Let me just do that.
