"""Fetch GitHub organization repositories with metadata using GitHub API."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import requests

GITHUB_API_URL = "https://api.github.com"


def fetch_repositories(
    organization: str = "nemardatasets",
    page_size: int = 100,
    timeout: float = 30.0,
    retries: int = 5,
    github_token: Optional[str] = None,
) -> Iterator[dict]:
    """Fetch all repositories from a GitHub organization.

    Args:
        organization: GitHub organization name
        page_size: Number of repositories per page (max 100)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        github_token: Optional GitHub personal access token for higher rate limits

    Yields:
        Repository metadata dictionaries

    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }

    # Add authentication if token provided (increases rate limit from 60 to 5000/hour)
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    page = 1

    while True:
        url = f"{GITHUB_API_URL}/orgs/{organization}/repos"
        params = {
            "per_page": min(page_size, 100),  # GitHub API max is 100
            "page": page,
            "type": "all",  # public, private, forks, sources, member, internal
            "sort": "created",  # created, updated, pushed, full_name
            "direction": "asc",  # asc or desc
        }

        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

                # Check rate limit
                if (
                    response.status_code == 403
                    and "rate limit" in response.text.lower()
                ):
                    print("  Warning: GitHub API rate limit exceeded")
                    print(
                        f"  Rate limit reset: {response.headers.get('X-RateLimit-Reset', 'unknown')}"
                    )
                    break

                response.raise_for_status()
                repos = response.json()

                # If empty list, we've fetched all repositories
                if not repos:
                    return

                for repo in repos:
                    repo_name = repo.get("name")

                    # Skip special GitHub repositories
                    if repo_name in [".github", ".gitignore"]:
                        continue

                    yield {
                        "dataset_id": repo_name,
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "url": repo.get("html_url"),
                        "created": repo.get("created_at"),
                        "modified": repo.get("updated_at"),
                        "pushed": repo.get("pushed_at"),
                        "size_kb": repo.get("size"),
                        "default_branch": repo.get("default_branch"),
                        "topics": repo.get("topics", []),
                        "language": repo.get("language"),
                        "has_wiki": repo.get("has_wiki"),
                        "has_issues": repo.get("has_issues"),
                        "archived": repo.get("archived"),
                        "visibility": repo.get("visibility"),
                        "clone_url": repo.get("clone_url"),
                        "ssh_url": repo.get("ssh_url"),
                    }

                # Check if there are more pages
                link_header = response.headers.get("Link", "")
                if 'rel="next"' not in link_header:
                    return

                page += 1
                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(
                    f"  Warning: Error fetching page {page} (attempt {attempt}/{retries}): {e}"
                )
                if attempt >= retries:
                    print(f"  Skipping to next page after {retries} failed attempts")
                    page += 1
                    break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all repositories from a GitHub organization with metadata."
    )
    parser.add_argument(
        "--organization",
        type=str,
        default="nemardatasets",
        help="GitHub organization name (default: nemardatasets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/github_organization_repos.json"),
        help="Output JSON file (default: consolidated/github_organization_repos.json).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Repositories per page (max 100, default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retry attempts (default: 5)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub personal access token (optional, increases rate limit)",
    )
    args = parser.parse_args()

    # Fetch and save
    print(f"Fetching repositories from GitHub organization: {args.organization}")
    repositories = list(
        fetch_repositories(
            organization=args.organization,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
            github_token=args.github_token,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(repositories, fh, indent=2)

    print(f"Saved {len(repositories)} repository entries to {args.output}")

    # Print summary
    if repositories:
        print("\nSummary:")
        print(f"  Organization: {args.organization}")
        print(f"  Total repositories: {len(repositories)}")

        # Count by language
        languages = {}
        for repo in repositories:
            lang = repo.get("language") or "None"
            languages[lang] = languages.get(lang, 0) + 1

        print(
            f"  Languages: {dict(sorted(languages.items(), key=lambda x: x[1], reverse=True))}"
        )

        # Count archived
        archived = sum(1 for repo in repositories if repo.get("archived"))
        print(f"  Archived: {archived}")

        # Count with topics
        with_topics = sum(1 for repo in repositories if repo.get("topics"))
        print(f"  With topics: {with_topics}")


if __name__ == "__main__":
    main()
