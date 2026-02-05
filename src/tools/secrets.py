import json
import logging
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
        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._secrets = {k: v for k, v in data.items() if isinstance(v, str)}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load secrets from %s: %s", self._path, e)

        logger.info("Loaded %d secret(s)", len(self._secrets))

    def _save(self) -> None:
        """Persist secrets to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._secrets, indent=2) + "\n")

    def list_secret_names(self) -> list[str]:
        """Return secret names only (never values)."""
        return sorted(self._secrets.keys())

    def get_secret(self, name: str) -> str | None:
        """Get a secret value. Used internally for env injection only."""
        return self._secrets.get(name)

    async def set_secret(self, name: str, value: str) -> str:
        """Create or update a secret."""
        self._secrets[name] = value
        self._save()
        return f"Secret '{name}' saved."

    async def delete_secret(self, name: str) -> str:
        """Delete a secret."""
        if name not in self._secrets:
            return f"Secret '{name}' not found."
        del self._secrets[name]
        self._save()
        return f"Secret '{name}' deleted."
