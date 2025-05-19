#!/bin/bash

# Check if input files are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <left_video> <right_video>"
    exit 1
fi

LEFT_VIDEO="$1"
RIGHT_VIDEO="$2"
OUTPUT="stacked_output.gif"

# Stack videos horizontally and add text overlays
ffmpeg -i "$LEFT_VIDEO" -i "$RIGHT_VIDEO" \
    -filter_complex "[0:v]format=yuv420p,scale=528:640[v0_scaled]; \
                    [1:v]format=yuv420p,scale=528:640[v1_scaled]; \
                    [v0_scaled]drawtext=text='Fallback':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=20[v0]; \
                    [v1_scaled]drawtext=text='CALF-TD3':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=20[v1]; \
                    [v0][v1]hstack=inputs=2,drawbox=x=528:y=0:w=2:h=640:color=white,split[a][b]; \
                    [a]palettegen=max_colors=256:stats_mode=single[p]; \
                    [b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[v]" \
    -map "[v]" \
    -t 8 \
    "$OUTPUT"

echo "Stacked GIF created as $OUTPUT" 