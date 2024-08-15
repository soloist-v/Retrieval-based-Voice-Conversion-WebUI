import onnx
from pathlib import Path

from onnxruntime import GraphOptimizationLevel
from onnxruntime.tools import optimize_onnx_model

# 加载 ONNX 模型
model_path = Path('test.onnx')
optimized_model_path = Path('test_optimized_model.onnx')
# onnx_model = onnx.load(model_path)

# 选择优化器
# passes = ['fuse_bn_into_conv', 'eliminate_identity', 'eliminate_deadend']

# 执行优化
optimized_model = optimize_onnx_model.optimize_model(
    model_path,
    optimized_model_path,
    GraphOptimizationLevel.ORT_DISABLE_ALL
)

# 保存优化后的模型
# onnx.save(optimized_model, optimized_model_path)
