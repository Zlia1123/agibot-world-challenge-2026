import torch

from src.evaluation.metrics import compute_metrics


def main() -> None:
    # Deutscher Kommentar:
    # Minimales Evaluationsskript mit Beispieldaten.
    pred = torch.randn(8, 32)
    target = torch.randn(8, 32)

    metrics = compute_metrics(pred, target)

    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
