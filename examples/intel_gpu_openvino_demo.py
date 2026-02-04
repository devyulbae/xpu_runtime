"""
Intel GPU inference with OpenVINO via xpuruntime.

Usage:
  pip install xpuruntime[openvino]
  pip install onnx   # optional, for the built-in demo model

  python examples/intel_gpu_openvino_demo.py [path_to_model.onnx]
  If no path is given, a small MLP (3-layer Gemm+ReLU) is created to exercise the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _make_demo_mlp_onnx_path() -> Path:
    """Create a small 3-layer MLP ONNX model (Gemm + ReLU) to exercise Intel GPU. Requires onnx."""
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        raise SystemExit(
            "For the built-in demo model, install onnx: pip install onnx\n"
            "Or run with a path to your own .onnx model: python intel_gpu_openvino_demo.py model.onnx"
        ) from None

    # Heavier MLP: [batch, 512] -> 1024 -> 512 -> 128, batch=64 to load the GPU
    batch = 64
    d_in, d1, d2, d_out = 512, 1024, 512, 128

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, d_in])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, d_out])

    np.random.seed(42)
    W1 = np.random.randn(d_in, d1).astype(np.float32) * 0.1
    b1 = np.zeros(d1, dtype=np.float32)
    W2 = np.random.randn(d1, d2).astype(np.float32) * 0.1
    b2 = np.zeros(d2, dtype=np.float32)
    W3 = np.random.randn(d2, d_out).astype(np.float32) * 0.1
    b3 = np.zeros(d_out, dtype=np.float32)

    init_W1 = numpy_helper.from_array(W1, "W1")
    init_b1 = numpy_helper.from_array(b1, "b1")
    init_W2 = numpy_helper.from_array(W2, "W2")
    init_b2 = numpy_helper.from_array(b2, "b2")
    init_W3 = numpy_helper.from_array(W3, "W3")
    init_b3 = numpy_helper.from_array(b3, "b3")

    # Gemm: Y = alpha*A*B + beta*C  ->  h1 = x*W1+b1, h2 = relu(h1)*W2+b2, y = relu(h2)*W3+b3
    gemm1 = helper.make_node("Gemm", ["x", "W1", "b1"], ["h1"], alpha=1.0, beta=1.0)
    relu1 = helper.make_node("Relu", ["h1"], ["h1r"])
    gemm2 = helper.make_node("Gemm", ["h1r", "W2", "b2"], ["h2"], alpha=1.0, beta=1.0)
    relu2 = helper.make_node("Relu", ["h2"], ["h2r"])
    gemm3 = helper.make_node("Gemm", ["h2r", "W3", "b3"], ["y"], alpha=1.0, beta=1.0)

    graph = helper.make_graph(
        [gemm1, relu1, gemm2, relu2, gemm3],
        "mlp",
        [X],
        [Y],
        initializer=[init_W1, init_b1, init_W2, init_b2, init_W3, init_b3],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)

    import tempfile
    p = Path(tempfile.gettempdir()) / "xpuruntime_demo_mlp.onnx"
    onnx.save(model, str(p))
    return p


def main() -> None:
    if len(sys.argv) >= 2:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            sys.exit(1)

    else:
        print("Creating demo MLP ONNX model (batch=64, 512->1024->512->128)...")
        model_path = _make_demo_mlp_onnx_path()
        print(f"Using: {model_path}")

    import numpy as np
    from xpuruntime import get_device_counts, format_device_counts
    from xpuruntime.inference import create_openvino_session

    print("\nDevice counts:")
    print(format_device_counts())

    # Session: device="auto" -> Intel GPU if available, else CPU
    # No OpenVINO Core/compile_model setup needed
    print("\nCreating OpenVINO session (device=auto -> Intel GPU when available)...")
    session = create_openvino_session(str(model_path), device="auto")
    print(f"Running on device: {session.device}")
    print(f"Input names: {session.input_names}")
    print(f"Output names: {session.output_names}")

    # Heavier workload: batch=64, run many iterations so GPU usage is visible in Task Manager
    inp_name = session.input_names[0]
    batch_size = 64
    d_in = 512
    inp = np.random.randn(batch_size, d_in).astype(np.float32) * 0.1

    import time
    n_iters = 500
    print(f"\nRunning {n_iters} iterations (batch={batch_size}, shape {inp.shape})...")
    t0 = time.perf_counter()
    for _ in range(n_iters):
        session.run({inp_name: inp})
    elapsed = time.perf_counter() - t0
    out = session.run({inp_name: inp})
    out = out[session.output_names[0]]
    print(f"Done in {elapsed:.2f}s ({n_iters / elapsed:.0f} iter/s)")
    print(f"Output shape: {out.shape}, first 3: {out.ravel()[:3].tolist()}")


if __name__ == "__main__":
    main()
