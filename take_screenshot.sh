#!/bin/bash

SCREENSHOT_DIR="screenshots"
FILENAME="$SCREENSHOT_DIR/$(date +%Y-%m-%d_%H-%M-%S).png"
SOCKET_PATH="/tmp/note_taker.sock"

## use scrot for linux
if screencapture -i "$FILENAME"; then
    ABSOLUTE_PATH="$(pwd)/$FILENAME"

    PAYLOAD=$(printf '{"command": "screenshot", "path": "%s"}' "$ABSOLUTE_PATH")

    echo "$PAYLOAD" | nc -U "$SOCKET_PATH"

    echo "Screenshot command sent for: $ABSOLUTE_PATH"
else
    echo "Screenshot canceled."
fi
