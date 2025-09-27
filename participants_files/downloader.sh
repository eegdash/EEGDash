#!/bin/bash

# Create the main directory if it doesn't exist
mkdir -p downloaded_files

# Check if links.txt exists
if [ ! -f "links.txt" ]; then
    echo "ERROR: The file 'links.txt' was not found in this directory."
    exit 1
fi

echo "Starting robust download process... 🤖"

while IFS= read -r url; do
  # Clean up the URL to remove any hidden characters
  clean_url=$(echo "$url" | tr -d '\r')

  echo "------------------------------------"
  echo "Processing URL: $clean_url"

  # Extract the dataset ID (e.g., 'ds005095')
  id=$(echo "$clean_url" | grep -o 'ds[0-9]*')

  if [ -z "$id" ]; then
    echo "--> WARNING: Could not extract a dataset ID. Skipping."
    continue
  fi

  echo "--> Extracted ID: $id"

  # Define the output directory and file path
  output_dir="downloaded_files/$id"
  output_file="${output_dir}/participants.tsv"
  mkdir -p "$output_dir"

  # --- NEW LOGIC STARTS HERE ---
  # First, check the HTTP status of the original URL without downloading
  http_code=$(curl -L -s -o /dev/null -w "%{http_code}" "$clean_url")

  if [ "$http_code" -eq 200 ]; then
    # If status is 200 OK, the URL is valid, so download it
    echo "--> Success (200 OK). Downloading file."
    curl -L -s -o "$output_file" "$clean_url"

  elif [ "$http_code" -eq 404 ]; then
    # If status is 404 Not Found, try switching from 'master' to 'main'
    echo "--> Failed (404 Not Found). Trying 'main' branch instead."
    main_url="${clean_url//\/master\//\/main\/}" # This replaces /master/ with /main/
    
    echo "--> New URL: $main_url"

    # Now, try downloading from the new 'main' URL
    curl -L -s -f -o "$output_file" "$main_url"

    # Check if the second download attempt succeeded
    if [ $? -eq 0 ]; then
      echo "--> Success with 'main' branch. Download complete."
    else
      echo "--> WARNING: 'main' branch also failed. Deleting empty file and skipping."
      rm -f "$output_file" # Clean up the empty file curl may have created on failure
    fi

  else
    # Handle other errors (e.g., 403 Forbidden, 500 Server Error)
    echo "--> WARNING: Request failed with HTTP status code: $http_code. Skipping."
  fi
  # --- NEW LOGIC ENDS HERE ---

done < links.txt

echo "------------------------------------"
echo "All downloads finished! ✅"