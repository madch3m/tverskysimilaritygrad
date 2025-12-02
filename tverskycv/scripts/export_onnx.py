import argparse
import torch
from pathlib import Path

# Use unified utilities
from tverskycv.training.config_utils import load_config
from tverskycv.training.model_builder import build_model_from_config
from tverskycv.training.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description="Export TverskyCV model to ONNX")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint .pt")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    ap.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export")
    ap.add_argument("--img-size", type=int, default=28, help="Dummy H=W for export (MNIST=28)")
    ap.add_argument("--in-ch", type=int, default=1, help="Input channels (MNIST=1, RGB=3)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    # Load config using unified utilities
    cfg = load_config(args.config)

    device = torch.device(args.device)
    model = build_model_from_config(cfg, device)
    model.eval()
    
    # Load checkpoint if provided
    if args.ckpt:
        load_checkpoint(args.ckpt, model, device)

    # Dummy input
    bs, c, h, w = args.batch_size, args.in_ch, args.img_size, args.img_size
    dummy = torch.randn(bs, c, h, w, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch"},
    }

    try:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"✅ Exported ONNX to: {out_path.resolve()}")
        print(f"   shape(input)={tuple(dummy.shape)}  opset={args.opset}")
    except Exception as e:
        print("❌ ONNX export failed.")
        print(f"Reason: {type(e).__name__}: {e}")
        print(
            "Hints:\n"
            "- Ensure the active head's forward is implemented (linear works; tversky may be stubbed).\n"
            "- Try a lower opset (e.g., --opset 13) if your environment lacks newer ops.\n"
            "- Export on CPU if CUDA export is problematic (--device cpu)."
        )


if __name__ == "__main__":
    main()