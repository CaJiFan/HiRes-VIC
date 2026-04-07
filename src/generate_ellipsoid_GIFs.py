import glob
import os
from PIL import Image

def create_episode_gif(episode, base_dir="./plots/ellipsoids", fps=5):
    """
    Finds all saved ellipsoid PNGs for a specific episode and 
    compiles them into an animated GIF.
    """
    # 1. Define the search pattern for this specific episode
    search_pattern = os.path.join(base_dir, f"ellipsoid_ep_{episode:02d}_step_*.png")
    
    # 2. Grab all matching files
    image_files = glob.glob(search_pattern)
    
    if not image_files:
        return  # Safely exit if no images were saved
        
    # 3. Sort files to ensure chronological order 
    # (Because you brilliantly padded with :04d, alphabetical sorting is flawless here)
    image_files.sort()
    
    # 4. Open all images as PIL Image objects
    frames = [Image.open(img_path) for img_path in image_files]
    
    # 5. Define output filename
    gif_filename = os.path.join(base_dir, f"animation_ep_{episode:02d}.gif")
    
    # Calculate duration per frame in milliseconds
    frame_duration = int(1000 / fps)
    
    # 6. Save as an infinitely looping GIF
    frames[0].save(
        gif_filename,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration,
        loop=0  # 0 forces it to loop continuously 
    )
    print(f"🎬 Saved Ellipsoid GIF for Episode {episode}: {gif_filename}")


if __name__ == "__main__":
    # Example usage: Generate GIFs for episodes 0 through 9
    for ep in range(10):
        create_episode_gif(ep)