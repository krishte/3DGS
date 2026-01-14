from core import Scene
from rendering import GaussianRasterizer
from training import train


def main():
    print("Initializing Gaussian Splatting training...")

    # Load scene
    scene = Scene("lego")  # Change to your object name
    print(f"Loaded scene with {len(scene.cameras)} cameras")
    print(f"Initialized {scene.gaussian_splats._num_splats} Gaussians")

    # Create rasterizer
    rasterizer = GaussianRasterizer()

    # Train
    train(scene, rasterizer, num_iterations=1000)  # Start with 1000 for testing

    print("Training complete!")


if __name__ == "__main__":
    main()
