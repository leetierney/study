#! /bin/bash

# Check if a directory name was provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 new_directory_name"
    exit 1
fi

# Define our new directory using the argument
new_dir=$1

# Create the new directory if it does not exist
if [ ! -d "$new_dir" ]; then
    mkdir "$new_dir"
fi

# Get the name of the script itself
script_name=$(basename "$0")

# Move all files from the current directory to the new directory except the script itself
for file in *; do
    if [ -f "$file" ] && [ "$file" != "$script_name" ]; then
        mv "$file" "$new_dir"
    fi
done

echo "All files except $script_name moved to $new_dir."