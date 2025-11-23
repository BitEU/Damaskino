#!/usr/bin/env python3
"""
Complete terrain pipeline: resample → extract tiles → generate C header
For WSEG-10 fallout simulation with embedded terrain data
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from shapely.geometry import Point
from pyproj import Transformer
import os
import struct

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
CSV_PATH = 'prepatory_CSVs/MASTER.csv'
INPUT_RASTER = 'rasters_SRTM15Plus/output_SRTM15Plus.asc'  # Source raster

# Output paths
RESAMPLED_DIR = 'rasters_SRTM15Plus_1000m'
RESAMPLED_RASTER = os.path.join(RESAMPLED_DIR, 'output_SRTM15Plus_1000m.asc')
TILES_DIR = 'extracted_tiles'
HEADER_FILE = 'terrain_data.h'

# Resampling settings
RESAMPLE_FACTOR = 2  # 500m → 1000m

# Tile extraction settings
RADIUS_KM = 100
SUBSAMPLE = 12       # 12 = ~17x17 grid for 100km radius
ELEV_MIN = -100      # Min elevation (m)
ELEV_MAX = 4500      # Max elevation (m)

# =============================================================================
# STEP 1: RESAMPLE 500m TO 1000m
# =============================================================================

def resample_raster():
    print("=" * 60)
    print("STEP 1: RESAMPLING 500m → 1000m")
    print("=" * 60)

    os.makedirs(RESAMPLED_DIR, exist_ok=True)

    with rasterio.open(INPUT_RASTER) as src:
        # Calculate new dimensions
        new_height = src.height // RESAMPLE_FACTOR
        new_width = src.width // RESAMPLE_FACTOR

        # Calculate new transform
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        # Read and resample
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.average
        )

        # Update profile
        profile = src.profile.copy()
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

        # Write resampled raster
        with rasterio.open(RESAMPLED_RASTER, 'w', **profile) as dst:
            dst.write(data, 1)

        print(f"  Input: {src.width}x{src.height} @ 500m")
        print(f"  Output: {new_width}x{new_height} @ 1000m")
        print(f"  Saved: {RESAMPLED_RASTER}")

    # Copy .prj file if exists
    prj_src = INPUT_RASTER.replace('.asc', '.prj')
    prj_dst = RESAMPLED_RASTER.replace('.asc', '.prj')
    if os.path.exists(prj_src):
        with open(prj_src, 'r') as f:
            prj_content = f.read()
        with open(prj_dst, 'w') as f:
            f.write(prj_content)
        print(f"  Copied: {prj_dst}")

    print()

# =============================================================================
# STEP 2: EXTRACT TILES FOR EACH LOCATION
# =============================================================================

def extract_tiles():
    print("=" * 60)
    print("STEP 2: EXTRACTING TILES (4-bit delta encoded)")
    print("=" * 60)

    os.makedirs(TILES_DIR, exist_ok=True)

    # Read locations
    df = pd.read_csv(CSV_PATH)
    print(f"  Found {len(df)} locations in CSV\n")

    with rasterio.open(RESAMPLED_RASTER) as src:
        print(f"  Raster bounds: {src.bounds}")
        print(f"  Raster shape: {src.height}x{src.width}\n")

        for idx, row in df.iterrows():
            location_name = row['Nuclear Target Name']
            lat = row['Latitude']
            lon = row['Longitude']
            location_id = row['ID']

            # Calculate UTM zone for accurate distance
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere = 'north' if lat >= 0 else 'south'
            utm_crs = f"EPSG:{32600 + utm_zone}" if hemisphere == 'north' else f"EPSG:{32700 + utm_zone}"

            # Transform to UTM, buffer, back to WGS84
            to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

            x_utm, y_utm = to_utm.transform(lon, lat)
            buffer_utm = Point(x_utm, y_utm).buffer(RADIUS_KM * 1000)

            minx_utm, miny_utm, maxx_utm, maxy_utm = buffer_utm.bounds
            min_lon, min_lat = to_wgs.transform(minx_utm, miny_utm)
            max_lon, max_lat = to_wgs.transform(maxx_utm, maxy_utm)

            # Clip to raster bounds
            min_lon = max(min_lon, src.bounds.left)
            max_lon = min(max_lon, src.bounds.right)
            min_lat = max(min_lat, src.bounds.bottom)
            max_lat = min(max_lat, src.bounds.top)

            # Check if within bounds
            if min_lon >= src.bounds.right or max_lon <= src.bounds.left or \
               min_lat >= src.bounds.top or max_lat <= src.bounds.bottom:
                print(f"  T{location_id:02d}: {location_name} - OUTSIDE BOUNDS, skipping")
                continue

            try:
                # Extract window
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                data = src.read(1, window=window)

                # Subsample
                data = data[::SUBSAMPLE, ::SUBSAMPLE]

                # Scale to 8-bit then to 4-bit
                data = np.clip(data, ELEV_MIN, ELEV_MAX)
                scaled = ((data - ELEV_MIN) / (ELEV_MAX - ELEV_MIN) * 255).astype(np.uint8)
                flat = ((scaled / 255.0) * 15).astype(np.uint8)

                # 4-bit delta encoding
                enc = bytearray()
                enc.append(flat.flat[0])

                for i in range(1, flat.size):
                    delta = int(flat.flat[i]) - int(flat.flat[i-1])
                    delta = max(-7, min(7, delta)) + 8
                    enc.append(delta)

                # Pack pairs into bytes
                packed = bytearray()
                for i in range(0, len(enc)-1, 2):
                    hi = enc[i] & 0x0F
                    lo = enc[i+1] & 0x0F if i+1 < len(enc) else 0
                    packed.append((hi << 4) | lo)
                if len(enc) % 2:
                    packed.append((enc[-1] & 0x0F) << 4)

                # Write tile
                output_path = os.path.join(TILES_DIR, f"T{location_id:02d}.BIN")
                with open(output_path, 'wb') as f:
                    f.write(struct.pack('<HHH', data.shape[1], data.shape[0], len(packed)))
                    f.write(packed)

                file_size = os.path.getsize(output_path)
                print(f"  T{location_id:02d}: {location_name[:20]:<20} {data.shape[1]}x{data.shape[0]} = {file_size} bytes")

            except Exception as e:
                print(f"  T{location_id:02d}: {location_name} - ERROR: {e}")

    # Summary
    total_bytes = sum(os.path.getsize(os.path.join(TILES_DIR, f))
                      for f in os.listdir(TILES_DIR) if f.endswith('.BIN'))
    print(f"\n  Total tile storage: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print()


def update_c_source(names):
    """Update FALLOUT_WSEG10.c so the LOCATION_NAMES array and menu bounds
    match the locations listed in the CSV (names is the list of display names).
    """
    c_path = 'FALLOUT_WSEG10.c'
    if not os.path.exists(c_path):
        raise FileNotFoundError(c_path)

    with open(c_path, 'r', encoding='utf-8') as f:
        src = f.read()

    # Build new block with a compile-time macro and the list (first entry is FLAT)
    n = len(names)
    entries = ['"FLAT (NO TERRAIN)"'] + [f'"{s.replace("\\", "\\\\").replace("\"", "\\\"")}"' for s in names]
    block_lines = []
    block_lines.append('// Location names for terrain tiles')
    block_lines.append(f'#define NUM_LOCATIONS {n}')
    block_lines.append(f'static const char* LOCATION_NAMES[NUM_LOCATIONS+1] = {{')
    for e in entries:
        block_lines.append(f'    {e},')
    block_lines.append('};')
    new_block = '\n'.join(block_lines) + '\n\n'

    # Replace existing LOCATION_NAMES block
    marker = 'Location names for terrain tiles'
    pos = src.find(marker)
    if pos == -1:
        raise RuntimeError('Could not find LOCATION_NAMES block marker in C source')

    decl_start = src.find('static const char* LOCATION_NAMES', pos)
    if decl_start == -1:
        raise RuntimeError('Could not find LOCATION_NAMES declaration in C source')

    decl_end = src.find('\n};', decl_start)
    if decl_end == -1:
        raise RuntimeError('Could not find end of LOCATION_NAMES block in C source')
    decl_end += 3  # include the closing '\n};'

    src = src[:decl_start] + new_block + src[decl_end:]

    # Update select menu loop, prompts and bounds (targeted replacements)
    src = src.replace('i<=35', 'i<=NUM_LOCATIONS')
    src = src.replace('SELECT LOCATION (0-35):', 'SELECT LOCATION (0-%d):')
    # Replace the printf that displayed the prompt, adding the macro param
    src = src.replace('printf("\nSELECT LOCATION (0-35): ");', 'printf("\nSELECT LOCATION (0-%d): ", NUM_LOCATIONS);')

    src = src.replace('loc>35', 'loc>NUM_LOCATIONS')
    src = src.replace('ERROR: Enter 0-35.\\n', 'ERROR: Enter 0-%d.\\n')
    src = src.replace('printf("ERROR: Enter 0-35.\\n");return 0;', 'printf("ERROR: Enter 0-%d.\\n", NUM_LOCATIONS);return 0;')

    # Also update the load_terrain bound check
    src = src.replace('if(loc<1||loc>35)return 0;', 'if(loc<1||loc>NUM_LOCATIONS)return 0;')

    with open(c_path, 'w', encoding='utf-8') as f:
        f.write(src)

# =============================================================================
# STEP 3: GENERATE C HEADER FILE
# =============================================================================

def generate_header():
    print("=" * 60)
    print("STEP 3: GENERATING C HEADER")
    print("=" * 60)

    with open(HEADER_FILE, 'w') as h:
        h.write("// Auto-generated terrain data - DO NOT EDIT\n")
        h.write("// 4-bit delta encoded elevation tiles\n")
        h.write("#ifndef TERRAIN_DATA_H\n#define TERRAIN_DATA_H\n\n")

        # Read CSV to determine which locations should be included and in what order
        df = pd.read_csv(CSV_PATH)
        ids = df['ID'].astype(int).tolist()
        names = df['Nuclear Target Name'].astype(str).tolist()

        # Collect all tile data (order follows CSV)
        offsets = []
        all_data = bytearray()

        for loc in ids:
            fname = os.path.join(TILES_DIR, f"T{loc:02d}.BIN")
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    data = f.read()
                offsets.append((loc, len(all_data), len(data)))
                all_data.extend(data)
            else:
                offsets.append((loc, 0, 0))

        # Write blob array
        h.write(f"static const unsigned char TERRAIN_BLOB[{len(all_data)}] = {{\n")
        for i in range(0, len(all_data), 16):
            chunk = all_data[i:i+16]
            h.write("  " + ",".join(f"0x{b:02x}" for b in chunk) + ",\n")
        h.write("};\n\n")

        # Write index - number of entries equals number of CSV rows
        idx_count = len(offsets)
        h.write(f"static const unsigned short TERRAIN_INDEX[{idx_count}][2] = {{\n")
        for loc, offset, length in offsets:
            h.write(f"  {{{offset},{length}}},  // T{loc:02d}\n")
        h.write("};\n\n")

        h.write("#endif\n")

    header_size = os.path.getsize(HEADER_FILE)
    print(f"  Generated: {HEADER_FILE}")
    print(f"  Terrain data: {len(all_data):,} bytes ({len(all_data)/1024:.1f} KB)")
    print(f"  Header file: {header_size:,} bytes ({header_size/1024:.1f} KB)")
    print()
    # Update C source to match CSV locations (names ordering)
    try:
        update_c_source(names)
        print(f"  Updated: FALLOUT_WSEG10.c (locations set to {len(names)})")
    except Exception as e:
        print(f"  WARNING: Could not update FALLOUT_WSEG10.c: {e}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  TERRAIN DATA BUILD PIPELINE")
    print("  For WSEG-10 Fallout Simulation")
    print("=" * 60 + "\n")

    # Check input files exist
    if not os.path.exists(INPUT_RASTER):
        print(f"ERROR: Input raster not found: {INPUT_RASTER}")
        exit(1)
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        exit(1)

    resample_raster()
    extract_tiles()
    generate_header()

    print("=" * 60)
    print("  BUILD COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  gcc -O2 -o fallout FALLOUT_WSEG10.c -lm")
    print()
