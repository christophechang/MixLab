from __future__ import annotations

import httpx

from mixlab.models import PlayedTrack

_TRACKS_PATH = "/api/catalog/tracks"
_MIXES_PATH = "/api/catalog/mixes"
_PAGE_SIZE = 100
_MIXES_PAGE_SIZE = 200


async def fetch_played_tracks(api_key: str = "", base_url: str = "") -> list[PlayedTrack]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    all_tracks: list[PlayedTrack] = []
    page = 1

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            response = await client.get(
                f"{base_url}{_TRACKS_PATH}",
                headers=headers,
                params={"page": page, "pageSize": _PAGE_SIZE},
            )

            if response.status_code != 200:
                raise RuntimeError(f"Catalog API returned {response.status_code}: {response.text}")

            data: dict[str, object] = response.json()
            items: list[dict[str, object]] = data.get("items", [])  # type: ignore[assignment]
            all_tracks.extend(
                PlayedTrack(artist=str(item.get("artist", "")), title=str(item.get("title", ""))) for item in items
            )

            raw_total = data.get("totalPages", 1)
            total_pages = int(raw_total) if isinstance(raw_total, (int, float, str)) else 1
            if page >= total_pages:
                break
            page += 1

    print(f"Fetched {len(all_tracks)} played tracks from API ({page} page(s)).", flush=True)
    return all_tracks


async def fetch_mix_names(api_key: str = "", base_url: str = "") -> list[str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{base_url}{_MIXES_PATH}",
            headers=headers,
            params={"page": 1, "pageSize": _MIXES_PAGE_SIZE},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Catalog API returned {response.status_code}: {response.text}")
        data: dict[str, object] = response.json()
        items: list[dict[str, object]] = data.get("items", [])  # type: ignore[assignment]
        return [str(item["title"]) for item in items if item.get("title")]
