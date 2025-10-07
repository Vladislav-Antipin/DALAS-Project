"""Base API client with caching and rate limiting."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.config import CACHE_DIR


class APIClient:
    """Base API client with caching support."""

    def __init__(self, base_url: str, cache_enabled: bool = True, rate_limit_delay: float = 0.5):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API
            cache_enabled: Whether to cache responses
            rate_limit_delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.cache_enabled = cache_enabled
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _get_cache_key(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key from endpoint and parameters."""
        cache_string = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return CACHE_DIR / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Load response from cache if it exists."""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """Save response to cache."""
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None, use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Make GET request with caching.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            use_cache: Whether to use cached response

        Returns:
            Response JSON data
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        # Make request
        url = f"{self.base_url}/{endpoint}"
        time.sleep(self.rate_limit_delay)  # Rate limiting

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, data)

        return data

    def batch_get(
        self, endpoints: list[str], params_list: list[dict[str, Any] | None] | None = None
    ) -> list[dict[str, Any]]:
        """
        Make multiple GET requests with progress bar.

        Args:
            endpoints: List of endpoints
            params_list: List of parameters for each endpoint (optional)

        Returns:
            List of response data
        """
        if params_list is None:
            params_list = [None] * len(endpoints)

        results = []
        for endpoint, params in tqdm(
            zip(endpoints, params_list, strict=True), total=len(endpoints)
        ):
            try:
                data = self.get(endpoint, params)
                results.append(data)
            except Exception as e:
                print(f"Error fetching {endpoint}: {e}")
                results.append(None)

        return results
