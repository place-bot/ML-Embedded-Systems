"""Optional static ONNX analysis (requires onnx); no inference or timing."""
import collections
import json
from pathlib import Path

import onnx


def inspect(path):
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(path))).graph
    weights = {item.name: list(item.dims) for item in graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(graph.input) + list(graph.value_info) + list(graph.output)}
    counts = collections.Counter(node.op_type for node in graph.node)
    grouped = 0
    conv_macs = 0
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        grouped += attrs.get("group", 1) > 1
        weight = weights[node.input[1]]
        output = shapes[node.output[0]]
        if not (len(weight) == len(output) == 4 and all(output)):
            raise ValueError("Expected known 2D convolution shapes")
        conv_macs += (output[0] * output[1] * output[2] * output[3]
                      * weight[1] * weight[2] * weight[3])
    return {"operator_counts": dict(counts), "grouped_convolutions": grouped,
            "convolution_macs_per_image": conv_macs,
            "note": "Static graph analysis, not runtime measurement. MAC count covers Conv nodes only."}


if __name__ == "__main__":
    models = Path(__file__).resolve().parents[1] / "models"
    print(json.dumps({p.stem: inspect(p) for p in sorted(models.glob("*.onnx"))}, indent=2))
