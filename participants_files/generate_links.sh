#!/bin/bash

# --- Configuration ---
# The file containing just the repository names (e.g., ds002718)
REPO_LIST_FILE="repo_list.txt"
# The GitHub owner/organization
OWNER="OpenNeuroDatasets"
# The file where we will save the final, correct URLs
OUTPUT_FILE="correct_links.txt"
# --- End of Configuration ---

# Clear the output file to start fresh
> "$OUTPUT_FILE"

echo "Querying GitHub API to find default branches... 🤖"

# Loop through each repository name in the list
while IFS= read -r repo_name; do
  # Construct the GitHub API URL for the repository
  api_url="https://api.github.com/repos/${OWNER}/${repo_name}"

  echo "Processing ${repo_name}..."

  # Use curl to call the API and jq to parse the default_branch name from the JSON response
  # The '-r' flag in jq removes the quotes from the output string
  default_branch=$(curl -s "$api_url" | jq -r .default_branch)

  # Check if we successfully got a branch name
  if [ "$default_branch" != "null" ] && [ -n "$default_branch" ]; then
    # Construct the final raw file URL using the correct branch
    file_url="https://raw.githubusercontent.com/${OWNER}/${repo_name}/${default_branch}/participants.tsv"
    echo "  --> Found branch: '${default_branch}'. URL created."
    # Append the correct URL to our output file
    echo "$file_url" >> "$OUTPUT_FILE"
  else
    echo "  --> ERROR: Could not find default branch for ${repo_name}. Repo may be private or deleted."
  fi
  sleep 1
done < "$REPO_LIST_FILE"

echo "------------------------------------"
echo "Finished! Your new list is ready in ${OUTPUT_FILE} ✅"