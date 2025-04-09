
# for displaying animation
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt

def display_frame(frame, processed_frame):
    """
    Display the original and preprocessed images side by side.
    Additionally, draw two red horizontal lines on the original image:
    - The first line is at row 34.
    - The second line is at row -16.

    Parameters:
    - frame: The original image/frame to be displayed.
    - preprocessed_frame: The preprocessed frame of the input frame.
    """
    print(f"Original Frame Shape: {frame.shape}")
    print(f"Processed Frame Shape: {processed_frame.shape}")

    # Draw the first red line on the original frame at row 34
    frame[34, :] = [255, 0, 0]
    # Draw the second red line on the original frame at row -16
    frame[-16, :] = [255, 0, 0]

    # Display the original image with red lines
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('Original Image (with red lines)')

    # Display the preprocessed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed_frame, cmap='Greys')
    plt.title('Preprocessed Image')

    # Show the images
    plt.show()


def save_animation(frames, filename="animation.mp4", fps=30, cmap="gray"):
    """
    Save a sequence of frames as an animation file.

    Args:
        frames (list): List of frames to animate.
        filename (str): Output filename (e.g., "animation.mp4").
        fps (int): Frames per second.
        cmap (str): Colormap to use for displaying frames.
    """
    plt.axis("off")
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], cmap=cmap, animated=True)

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 // fps,
        blit=True
    )
    ani.save(filename, fps=fps, writer="ffmpeg")
    print(f"Animation saved to {filename}")


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.title("Scores")
    plt.plot(scores)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    return rolling_mean