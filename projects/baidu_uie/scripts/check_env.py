"""环境自检：对齐文档 §5.3 的轻量版（本机 Windows / CPU 友好）。"""

from __future__ import annotations


def main() -> None:
    print("=== UIE env check ===")
    try:
        import charset_normalizer
        import paddlenlp
        import paddle

        print("IMPORT_OK", "paddlenlp", paddlenlp.__version__, "paddle", paddle.__version__)
    except Exception as e:
        print("IMPORT_FAIL", e)
        raise SystemExit(1)

    print("cuda_compiled", paddle.is_compiled_with_cuda())
    try:
        device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.device.set_device(device)
        x = paddle.randn([2, 4])
        y = paddle.matmul(x, x.T)
        print("PADDLE_OK", y.shape, "device", device)
    except Exception as e:
        print("PADDLE_FAIL", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
