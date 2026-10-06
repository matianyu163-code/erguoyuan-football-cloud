"""Local OpenFootball provider pinned to a verified Git commit."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenFootballSourceFile:
    path: Path
    repository: str
    commit_sha: str
    dataset_path: str
    content_hash: str
    license_id: str = "CC0-1.0"


class OpenFootballProvider:
    """Verify local source bytes against the exact tracked Git blob."""

    REPOSITORY = "https://github.com/openfootball/football.json"
    PINNED_COMMIT = "e6744429ee395bc86f247348c6184bb08d4eb361"

    def __init__(self, repository_path: Path):
        self.repository_path = repository_path.resolve()
        if not (self.repository_path / ".git").exists():
            raise ValueError("PINNED_LOCAL_REPOSITORY_MISSING")

    def _git(self, *args: str) -> bytes:
        result = subprocess.run(["git", "-c", f"safe.directory={self.repository_path.as_posix()}",
                                 *args], cwd=self.repository_path, capture_output=True, check=True)
        return result.stdout

    @property
    def commit_sha(self) -> str:
        commit = self._git("rev-parse", "HEAD").decode("ascii").strip()
        if commit != self.PINNED_COMMIT:
            raise ValueError("OPENFOOTBALL_COMMIT_NOT_APPROVED")
        return commit

    def source_file(self, season: str, code: str) -> OpenFootballSourceFile:
        dataset_path = f"{season}/{code}.json"
        path = self.repository_path / dataset_path
        if not path.is_file():
            raise FileNotFoundError(dataset_path)
        commit = self.commit_sha
        tracked = self._git("show", f"{commit}:{dataset_path}")
        local = path.read_bytes()
        # Git's Windows checkout may expand line endings; compare the tracked
        # canonical blob after reversing that transport-only transformation.
        normalized_local = local.replace(b"\r\n", b"\n")
        tracked_hash = hashlib.sha256(tracked).digest()
        if tracked_hash not in {hashlib.sha256(local).digest(), hashlib.sha256(normalized_local).digest()}:
            raise ValueError(f"PINNED_SOURCE_CONTENT_MISMATCH:{dataset_path}")
        return OpenFootballSourceFile(path, self.REPOSITORY, commit, dataset_path,
                                      hashlib.sha256(local).hexdigest())
