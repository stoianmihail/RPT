#!/bin/bash

# Process all SQL files in the current directory
for file in *.sql; do
    # Create a cleaned version of the file
    sed -E '/^--|^:[a-zA-Z0-9_-]+|^\s*$/d' "$file" > "${file}.clean"
    mv "${file}.clean" "$file"
    echo "Cleaned $file"
done

echo "All SQL files have been cleaned."

