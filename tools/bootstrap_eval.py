"""Bootstrap confidence intervals for accuracy from inference results JSON."""

import argparse
import json
import numpy as np


def bootstrap_accuracy(accuracies, n_bootstrap=10000, ci=95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(accuracies, size=n, replace=True)
        bootstrap_means[i] = sample.mean()
    alpha = (100 - ci) / 2
    ci_low = np.percentile(bootstrap_means, alpha)
    ci_high = np.percentile(bootstrap_means, 100 - alpha)
    return bootstrap_means, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Bootstrap accuracy CI from inference results JSON")
    parser.add_argument("json_path", help="Path to inference_results_*.json")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap iterations (default: 10000)")
    parser.add_argument("--ci", type=float, default=95, help="Confidence interval percentage (default: 95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.json_path) as f:
        data = json.load(f)

    results = data["results"]
    accuracies = [r["accuracy"] for r in results]
    n = len(accuracies)
    original_acc = np.mean(accuracies)
    n_correct = sum(accuracies)

    bootstrap_means, ci_low, ci_high = bootstrap_accuracy(accuracies, args.n_bootstrap, args.ci)

    print(f"Original accuracy:  {original_acc:.4f}  ({int(n_correct)}/{n})")
    print(f"Bootstrap mean:     {bootstrap_means.mean():.4f}")
    print(f"Bootstrap std:      {bootstrap_means.std():.4f}")
    print(f"{args.ci:.0f}% CI:            [{ci_low:.4f}, {ci_high:.4f}]")

    # Per-type breakdown if multiple question types exist
    types = set(r.get("question_type") for r in results if r.get("question_type"))
    if len(types) > 1:
        print("\nPer-type breakdown:")
        for qt in sorted(types):
            qt_accs = [r["accuracy"] for r in results if r.get("question_type") == qt]
            _, low, high = bootstrap_accuracy(qt_accs, args.n_bootstrap, args.ci)
            print(f"  {qt}: {np.mean(qt_accs):.4f}  (n={len(qt_accs)})  {args.ci:.0f}% CI [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
