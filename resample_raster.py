"""
Resample topographical raster data from 500m to 1000m resolution.

This script reads raster files from the rasters_SRTM15Plus directory and
downsamples them from 500m to 1000m resolution using bilinear interpolation.
"""

import os
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def resample_raster(input_file, output_file, scale_factor=0.5):
    """
    Resample a raster file to a lower resolution.

    Args:
        input_file: Path to input raster file
        output_file: Path to output raster file
        scale_factor: Resolution scaling factor (0.5 = half resolution = double pixel size)
    """
    with rasterio.open(input_file) as src:
        # Get original dimensions
        original_width = src.width
        original_height = src.height

        # Calculate new dimensions (half the size for 2x pixel size)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Get the transform for the new resolution
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Copy metadata and update for new dimensions
        metadata = src.meta.copy()
        metadata.update({
            'width': new_width,
            'height': new_height,
            'transform': transform
        })

        print(f"Input resolution: {original_width}x{original_height}")
        print(f"Output resolution: {new_width}x{new_height}")
        print(f"Pixel size change: {src.res[0]:.1f}m -> {src.res[0]/scale_factor:.1f}m")

        # Create output file and resample
        with rasterio.open(output_file, 'w', **metadata) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear  # Good for elevation data
                )

        print(f"Successfully resampled to {output_file}")


def main():
    # Define paths
    input_dir = Path("rasters_SRTM15Plus")
    output_dir = Path("rasters_SRTM15Plus_1000m")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all raster files (.asc, .tif, .tiff)
    raster_extensions = ['.asc', '.tif', '.tiff']
    raster_files = []
    for ext in raster_extensions:
        raster_files.extend(input_dir.glob(f'*{ext}'))

    if not raster_files:
        print(f"No raster files found in {input_dir}")
        return

    print(f"Found {len(raster_files)} raster file(s) to process\n")

    # Process each raster file
    for input_file in raster_files:
        print(f"\nProcessing: {input_file.name}")
        print("-" * 60)

        # Create output filename
        output_file = output_dir / f"{input_file.stem}_1000m{input_file.suffix}"

        try:
            # Resample from 500m to 1000m (scale_factor = 0.5)
            resample_raster(str(input_file), str(output_file), scale_factor=0.5)
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Resampled files saved to: {output_dir}")


if __name__ == "__main__":
    main()
