"""Visualization utilities for Project Atlas."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_training_curves(history: dict, save_path: Optional[str] = None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Project Atlas — Training Progress", fontsize=14, fontweight="bold")
    
    # Total loss
    if "train_loss" in history and history["train_loss"]:
        axes[0, 0].plot(history["train_loss"], color="#2196F3", linewidth=1.5)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    if "recon_loss" in history and history["recon_loss"]:
        axes[0, 1].plot(history["recon_loss"], color="#4CAF50", linewidth=1.5)
        axes[0, 1].set_title("Reconstruction Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    if "kl_loss" in history and history["kl_loss"]:
        axes[1, 0].plot(history["kl_loss"], color="#FF9800", linewidth=1.5)
        axes[1, 0].set_title("KL Divergence")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Surprise rate
    if "surprise_rate" in history and history["surprise_rate"]:
        axes[1, 1].plot(history["surprise_rate"], color="#E91E63", linewidth=1.5)
        axes[1, 1].set_title("Surprise Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Rate")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved training curves to {save_path}")
    
    plt.close()
    return fig


def plot_latent_space(latent_states: np.ndarray, labels: Optional[np.ndarray] = None, 
                       save_path: Optional[str] = None):
    """Plot latent space using PCA/t-SNE."""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(latent_states)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if labels is not None:
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, 
                           cmap="viridis", alpha=0.6, s=10)
        plt.colorbar(scatter)
    else:
        ax.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=10, color="#2196F3")
    
    ax.set_title("Latent Space (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
    return fig


def plot_imagined_trajectory(real_trajectory: np.ndarray, imagined_trajectory: np.ndarray,
                              save_path: Optional[str] = None):
    """Compare real vs imagined trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real trajectory
    axes[0].plot(real_trajectory[:, 0], real_trajectory[:, 1], 
                "o-", color="#4CAF50", markersize=3, linewidth=1, label="Real")
    axes[0].set_title("Real Trajectory")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Imagined trajectory
    axes[1].plot(imagined_trajectory[:, 0], imagined_trajectory[:, 1],
                "o-", color="#E91E63", markersize=3, linewidth=1, label="Imagined")
    axes[1].set_title("Imagined Trajectory")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle("Real vs Imagined", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
    return fig


def create_animation(frames: list[np.ndarray], save_path: str, fps: int = 10):
    """Create a GIF animation from frames."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx])
        ax.set_title(f"Step {frame_idx}")
        ax.axis("off")
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()
    
    print(f"🎬 Saved animation to {save_path}")
