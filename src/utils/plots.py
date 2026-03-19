
# src/visualization/plots.py

from pathlib import Path
import matplotlib.pyplot as plt

def save_roc_curve(fpr, tpr, auc_value: float, out_path: Path, title: str):
    """
    Función creada a partir del code de Dani. Guarda una curva ROC simple.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()