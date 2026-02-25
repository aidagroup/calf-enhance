#!/bin/bash
set -euo pipefail

# Check if input files are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <left_video> <right_video>"
    exit 1
fi

LEFT_VIDEO="$1"
RIGHT_VIDEO="$2"
OUTPUT="stacked_output.gif"

# Stack videos horizontally without stretching: fit into square and pad.
ffmpeg -y -i "$LEFT_VIDEO" -i "$RIGHT_VIDEO" \
    -filter_complex "[0:v]format=yuv420p,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black[v0_scaled]; \
                    [1:v]format=yuv420p,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black[v1_scaled]; \
                    [v0_scaled]drawtext=text='Baseline policy':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=20[v0]; \
                    [v1_scaled]drawtext=text='Enhanced Policy':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=20[v1]; \
                    [v0][v1]hstack=inputs=2,drawbox=x=640:y=0:w=2:h=640:color=white,split[a][b]; \
                    [a]palettegen=max_colors=256:stats_mode=single[p]; \
                    [b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[v]" \
    -map "[v]" \
    -t 8 \
    "$OUTPUT"

echo "Stacked GIF created as $OUTPUT"
