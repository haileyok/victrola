import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretManager:
    """Manages secrets stored locally as a JSON file.

    Secrets are name -> value pairs. The agent can see secret names but never
    their values. Values are injected as environment variables into Deno
    custom tool processes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._secrets: dict[str, str] = {}

    async def load_secrets(self) -> None:
        """Load secrets from the local JSON file."""
        self._secrets.clear()

        def _read():
            if not self._path.exists():
                return None
            return self._path.read_text()

        try:
            content = await asyncio.to_thread(_read)
            if content is None:
                return
            data = json.loads(content)
            if isinstance(data, dict):
                self._secrets = {k: v for k, v in data.items() if isinstance(v, str)}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load secrets from %s: %s", self._path, e)

        logger.info("Loaded %d secret(s)", len(self._secrets))

    async def _save(self) -> None:
        """Persist secrets to disk atomically (tempfile + os.replace)."""
        def _write():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to temp file then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json.dumps(self._secrets, indent=2) + "\n")
                os.replace(tmp_path, str(self._path))
            except Exception:
                # Clean up the temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        await asyncio.to_thread(_write)

    def list_secret_names(self) -> list[str]:
        """Return secret names only (never values)."""
        return sorted(self._secrets.keys())

    def get_secret(self, name: str) -> str | None:
        """Get a secret value. Used internally for env injection only."""
        return self._secrets.get(name)

    async def set_secret(self, name: str, value: str) -> str:
        """Create or update a secret."""
        self._secrets[name] = value
        await self._save()
        return f"Secret '{name}' saved."

    async def delete_secret(self, name: str) -> str:
        """Delete a secret."""
        if name not in self._secrets:
            return f"Secret '{name}' not found."
        del self._secrets[name]
        await self._save()
        return f"Secret '{name}' deleted."
