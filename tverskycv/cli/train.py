# tverskycv/cli/train.py
import argparse

from tverskycv.training.entry_points import train_from_config


def main():
    parser = argparse.ArgumentParser(description="Train a TverskyCV model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Override device from config")
    parser.add_argument("--use-optimized", action="store_true", help="Use OptimizedTrainer")
    args = parser.parse_args()

    # Use unified training function
    results = train_from_config(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        device=args.device,
        use_optimized_trainer=args.use_optimized
    )

    print(f"\n✓ Training complete!")
    print(f"Best validation accuracy: {results.get('best_val_acc', 0.0):.4f}")
    if 'best_epoch' in results:
        print(f"Best epoch: {results['best_epoch']}")


if __name__ == "__main__":
    main()
