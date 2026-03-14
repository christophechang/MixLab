from __future__ import annotations

import pytest
import respx
from httpx import Response

from mixlab.client import fetch_played_tracks

_BASE = "https://api.changsta.com"
_URL = f"{_BASE}/api/catalog/tracks"

_PAGE1 = {
    "items": [{"artist": "Calibre", "title": "All Good"}, {"artist": "Noisia", "title": "Stigma"}],
    "totalPages": 2,
}
_PAGE2 = {
    "items": [{"artist": "Skeptical", "title": "Residual"}],
    "totalPages": 2,
}


@respx.mock
async def test_fetch_played_tracks_happy_path_single_page() -> None:
    respx.get(_URL).mock(return_value=Response(200, json={**_PAGE1, "totalPages": 1}))
    tracks = await fetch_played_tracks("test-key", _BASE)
    assert len(tracks) == 2
    assert tracks[0].artist == "Calibre"
    assert tracks[1].title == "Stigma"


@respx.mock
async def test_fetch_played_tracks_paginates_all_pages() -> None:
    route = respx.get(_URL).mock(side_effect=[Response(200, json=_PAGE1), Response(200, json=_PAGE2)])
    tracks = await fetch_played_tracks("test-key", _BASE)
    assert len(tracks) == 3
    assert tracks[2].artist == "Skeptical"
    assert route.call_count == 2


@respx.mock
async def test_fetch_played_tracks_raises_on_401() -> None:
    respx.get(_URL).mock(return_value=Response(401, text="Unauthorized"))
    with pytest.raises(RuntimeError, match="401"):
        await fetch_played_tracks("bad-key", _BASE)


@respx.mock
async def test_fetch_played_tracks_raises_on_500() -> None:
    respx.get(_URL).mock(return_value=Response(500, text="Internal Server Error"))
    with pytest.raises(RuntimeError, match="500"):
        await fetch_played_tracks("test-key", _BASE)
