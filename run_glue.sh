#!/bin/bash
# Wrapper script: Use caffeinate to prevent MacBook from sleeping

# caffeinate options:
# -i: Prevent system idle sleep (keeps running even when screen is off)
# -d: Prevent display sleep (optional, omit if you only want to prevent system sleep)
# -w <PID>: Wait for specified process to finish

echo "=== Using caffeinate to prevent MacBook from sleeping ==="
echo "During script execution, MacBook will not sleep even when screen is off"
echo "Press Ctrl+C to stop at any time"
echo ""

caffeinate -i bash glue.sh

echo ""
echo "=== Training completed, caffeinate stopped ==="

