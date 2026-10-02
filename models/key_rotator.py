"""
Multi-account API key rotation and health tracking engine.
Allows seamless rotation across multiple accounts for cloud providers (Groq, Gemini, OpenRouter),
with independent rate-limit (RPM/TPM), cooldown, and daily quota accounting per account.
"""
import collections
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from logger import get_logger

logger = get_logger("rotator")


class KeyStatus:
    HEALTHY = "healthy"
    COOLDOWN = "cooldown"      # Temporary 429 (RPM / TPM window)
    EXHAUSTED = "exhausted"    # Daily limit reached (TPD / RPD)
    DISABLED = "disabled"      # Auth failed (401 / 403 invalid key)


class AccountKey:
    """Represents a single API key from an account with independent health and rate-limit tracking."""

    def __init__(self, key: str, index: int, provider: str = "generic"):
        self.key: str = key.strip()
        self.index: int = index
        self.provider: str = provider
        self.status: str = KeyStatus.HEALTHY
        self.cooldown_until: float = 0.0
        self.exhausted_until: float = 0.0
        self.last_used: float = 0.0

        # Performance and usage counters
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.consecutive_errors: int = 0
        self.tokens_used: int = 0

        # Per-account TPM tracking: model -> deque of [timestamp, tokens]
        self._tpm_usage: Dict[str, collections.deque] = {}
        self._model_tpm_limits: Dict[str, int] = {}

    @property
    def masked_key(self) -> str:
        """Safe representation for logs and UI display."""
        if not self.key:
            return ""
        if len(self.key) <= 8:
            return f"***{self.key[-3:]}"
        return f"{self.key[:5]}...{self.key[-4:]}"

    @property
    def account_id(self) -> str:
        return f"{self.provider.capitalize()} Account #{self.index + 1} ({self.masked_key})"

    def is_available(self, now: Optional[float] = None) -> bool:
        """Check if this account key is healthy and ready for requests."""
        if self.status == KeyStatus.DISABLED:
            return False

        if now is None:
            now = time.time()

        if self.status == KeyStatus.EXHAUSTED:
            if now < self.exhausted_until:
                return False
            # Daily reset has elapsed
            self.status = KeyStatus.HEALTHY
            self.exhausted_until = 0.0
            logger.info("[%s] %s daily quota reset has elapsed; restored to HEALTHY.", self.provider, self.account_id)

        if self.status == KeyStatus.COOLDOWN:
            if now < self.cooldown_until:
                return False
            # Cooldown has elapsed
            self.status = KeyStatus.HEALTHY
            self.cooldown_until = 0.0
            logger.info("[%s] %s cooldown period elapsed; restored to HEALTHY.", self.provider, self.account_id)

        return True

    def seconds_until_available(self, now: Optional[float] = None) -> float:
        """Returns seconds until key becomes available, or 0.0 if already available."""
        if self.status == KeyStatus.DISABLED:
            return float("inf")

        if now is None:
            now = time.time()

        wait = 0.0
        if self.status == KeyStatus.EXHAUSTED and now < self.exhausted_until:
            wait = max(wait, self.exhausted_until - now)
        if self.status == KeyStatus.COOLDOWN and now < self.cooldown_until:
            wait = max(wait, self.cooldown_until - now)
        return wait

    def record_request(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        self.last_used = now
        self.total_requests += 1

    def record_success(self, tokens: int = 0) -> None:
        self.successful_requests += 1
        self.consecutive_errors = 0
        try:
            tokens_val = int(tokens or 0)
        except (TypeError, ValueError):
            tokens_val = 0
        self.tokens_used += max(0, tokens_val)
        if self.status in (KeyStatus.COOLDOWN, KeyStatus.EXHAUSTED):
            self.status = KeyStatus.HEALTHY
            self.cooldown_until = 0.0
            self.exhausted_until = 0.0

    def record_rate_limit(
        self,
        retry_after: float,
        is_daily: bool = False,
        header_limit: Optional[int] = None,
        model: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.time()

        self.failed_requests += 1
        self.consecutive_errors += 1

        if model and header_limit and header_limit > 0:
            self._model_tpm_limits[model] = header_limit

        try:
            safe_retry = float(retry_after or 1.0)
        except (TypeError, ValueError):
            safe_retry = 1.0

        if is_daily:
            self.status = KeyStatus.EXHAUSTED
            self.exhausted_until = now + max(safe_retry, 60.0)
            logger.warning(
                "[%s] %s reached DAILY quota exhaustion; marked EXHAUSTED until +%.0fs.",
                self.provider, self.account_id, max(safe_retry, 60.0),
            )
        else:
            self.status = KeyStatus.COOLDOWN
            self.cooldown_until = now + max(safe_retry, 1.0)
            logger.warning(
                "[%s] %s rate-limited (429); marked COOLDOWN for %.1fs.",
                self.provider, self.account_id, safe_retry,
            )

    def record_error(self, is_auth_error: bool = False) -> None:
        self.failed_requests += 1
        self.consecutive_errors += 1
        if is_auth_error:
            self.status = KeyStatus.DISABLED
            logger.error("[%s] %s authentication failed (401/403); key DISABLED.", self.provider, self.account_id)

    # ─── Per-Account TPM Budgeting ─────────────────────────────────────────────
    def get_tpm_limit(self, model: str, default_limit: int) -> int:
        return self._model_tpm_limits.get(model, default_limit)

    def prune_tpm(self, model: str, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        dq = self._tpm_usage.get(model)
        if not dq:
            return
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def check_tpm_budget(
        self, model: str, estimated_tokens: int, limit: int, window_seconds: float, safety_margin: float, now: float,
    ) -> Tuple[bool, float, Optional[List[Any]]]:
        """
        Check if this account has enough TPM budget for the given estimate.
        Returns: (has_budget, wait_time, reservation)
        """
        self.prune_tpm(model, now, window_seconds)
        budget = limit * (1.0 - safety_margin)
        dq = self._tpm_usage.setdefault(model, collections.deque())
        current_sum = sum(tokens for _, tokens in dq)

        if current_sum + estimated_tokens <= budget:
            reservation = [now, estimated_tokens]
            dq.append(reservation)
            return True, 0.0, reservation

        wait = max(0.5, dq[0][0] + window_seconds - now) if dq else 0.5
        return False, wait, None

    def settle_tpm(self, reservation: Optional[List[Any]], actual_tokens: float) -> None:
        if reservation is not None:
            try:
                tokens_flt = float(actual_tokens or 0.0)
            except (TypeError, ValueError):
                tokens_flt = 0.0
            reservation[1] = max(0.0, tokens_flt)

    def get_window_tokens(self, model: Optional[str] = None, now: Optional[float] = None, window_seconds: float = 60.0) -> int:
        """Calculate total tokens consumed by this account in the rolling window."""
        if now is None:
            now = time.time()
        cutoff = now - window_seconds
        total = 0
        for m, dq in self._tpm_usage.items():
            if model and m != model:
                continue
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            total += sum(tokens for _, tokens in dq)
        return total

    def get_remaining_headroom(
        self,
        model: str,
        default_limit: int = 6000,
        now: Optional[float] = None,
        window_seconds: float = 60.0,
        safety_margin: float = 0.20,
    ) -> int:
        """Returns estimated available token headroom for this account within the current sliding window."""
        if now is None:
            now = time.time()
        limit = self.get_tpm_limit(model, default_limit)
        budget = int(limit * (1.0 - safety_margin))
        used = self.get_window_tokens(model=model, now=now, window_seconds=window_seconds)
        return max(0, budget - used)

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        # Ensure status reflects current time
        self.is_available(now)
        return {
            "index": self.index,
            "masked_key": self.masked_key,
            "status": self.status,
            "account_id": self.account_id,
            "is_available": self.is_available(now),
            "seconds_until_available": round(self.seconds_until_available(now), 1),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "consecutive_errors": self.consecutive_errors,
            "tokens_used": self.tokens_used,
            "window_tokens_60s": self.get_window_tokens(now=now),
            "last_used_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_used)) if self.last_used else None,
        }


class KeyRotator:
    """
    Thread-safe multi-account API key rotator for a single provider.
    Distributes calls across accounts and fails over immediately when an account hits rate limits.
    """

    def __init__(
        self,
        provider: str,
        keys: Optional[List[str]] = None,
        strategy: str = "least_loaded",
    ):
        self.provider: str = provider.lower()
        self.strategy: str = strategy.lower()
        self._lock = threading.RLock()
        self._accounts: List[AccountKey] = []
        self._key_map: Dict[str, AccountKey] = {}
        self._current_idx: int = 0

        if keys:
            self.set_keys(keys)

    def set_keys(self, keys: Any) -> None:
        """Update the key pool, preserving historical statistics of existing keys."""
        with self._lock:
            if isinstance(keys, str):
                raw_list = [k.strip() for k in keys.split(",") if k.strip()]
            elif isinstance(keys, (list, tuple, set)):
                raw_list = [str(k).strip() for k in keys if str(k).strip()]
            else:
                raw_list = []

            # Deduplicate while preserving order
            seen = set()
            clean_keys = []
            for k in raw_list:
                if k not in seen:
                    seen.add(k)
                    clean_keys.append(k)

            old_map = self._key_map
            new_accounts = []
            new_map = {}

            for idx, key in enumerate(clean_keys):
                if key in old_map:
                    account = old_map[key]
                    account.index = idx
                else:
                    account = AccountKey(key=key, index=idx, provider=self.provider)
                new_accounts.append(account)
                new_map[key] = account

            self._accounts = new_accounts
            self._key_map = new_map
            clean_keys_set = set(clean_keys)
            old_keys_set = set(old_map.keys())
            if clean_keys and clean_keys_set != old_keys_set:
                self._current_idx = 0
            elif self._current_idx >= len(self._accounts):
                self._current_idx = 0

            if self._accounts:
                logger.info(
                    "[%s] KeyRotator loaded %d account key(s) (strategy: %s).",
                    self.provider, len(self._accounts), self.strategy,
                )

    def get_keys(self) -> List[str]:
        with self._lock:
            return [acc.key for acc in self._accounts]

    def get_account_by_key(self, key: str) -> Optional[AccountKey]:
        with self._lock:
            return self._key_map.get(key)

    def get_active_account(
        self,
        proactive_rotate: bool = False,
        model: Optional[str] = None,
        estimated_tokens: int = 0,
    ) -> Optional[AccountKey]:
        """
        Selects the best available healthy account according to the rotation strategy.
        Prioritizes the key that has used the fewest tokens / has the most headroom in the current window.
        """
        with self._lock:
            if not self._accounts:
                return None

            now = time.time()
            healthy_accounts = [acc for acc in self._accounts if acc.is_available(now)]

            if not healthy_accounts:
                # All accounts are either cooling down or exhausted
                return None

            if proactive_rotate and len(healthy_accounts) > 1:
                # Advance pointer
                self._current_idx = (self._current_idx + 1) % len(self._accounts)

            n = len(self._accounts)

            if self.strategy == "least_loaded":
                # Primary strategy: pick the healthy account with the fewest window tokens (most immediate headroom).
                # Break ties using cumulative tokens used, then longest rested, then cyclic distance from current index.
                if estimated_tokens and estimated_tokens > 0:
                    with_headroom = [
                        acc for acc in healthy_accounts
                        if acc.get_remaining_headroom(model=model or "default", now=now) >= estimated_tokens
                    ]
                    candidates = with_headroom if with_headroom else healthy_accounts
                else:
                    candidates = healthy_accounts

                best = min(
                    candidates,
                    key=lambda acc: (
                        acc.get_window_tokens(model=model, now=now),
                        acc.tokens_used or 0,
                        acc.last_used or 0.0,
                        (acc.index - self._current_idx) % n,
                    ),
                )
                self._current_idx = best.index
                return best

            if self.strategy == "least_recently_used":
                # Pick the healthy account that hasn't been used the longest
                best = min(healthy_accounts, key=lambda acc: acc.last_used)
                self._current_idx = best.index
                return best

            # Default: round_robin
            # Scan starting from _current_idx for the first available healthy key
            n = len(self._accounts)
            for step in range(n):
                candidate_idx = (self._current_idx + step) % n
                candidate = self._accounts[candidate_idx]
                if candidate.is_available(now):
                    self._current_idx = candidate_idx
                    return candidate

            return healthy_accounts[0]

    def get_key(
        self,
        proactive_rotate: bool = False,
        model: Optional[str] = None,
        estimated_tokens: int = 0,
    ) -> str:
        """Returns the raw key string of the best available account."""
        acc = self.get_active_account(
            proactive_rotate=proactive_rotate,
            model=model,
            estimated_tokens=estimated_tokens,
        )
        if acc:
            return acc.key
        with self._lock:
            if self._accounts:
                # Even if cooling down, return current key as fallback
                return self._accounts[self._current_idx % len(self._accounts)].key
        return ""

    def rotate(self) -> bool:
        """Manually or periodically step to the next healthy account."""
        with self._lock:
            if not self._accounts or len(self._accounts) < 2:
                return False

            now = time.time()
            n = len(self._accounts)
            for step in range(1, n + 1):
                candidate_idx = (self._current_idx + step) % n
                candidate = self._accounts[candidate_idx]
                if candidate.is_available(now):
                    self._current_idx = candidate_idx
                    logger.info("[%s] Switched to %s.", self.provider, candidate.account_id)
                    return True

            # If no healthy keys found, still advance the index so we don't hammer only one
            self._current_idx = (self._current_idx + 1) % n
            logger.info(
                "[%s] Rotated pointer to %s (in cooldown/exhausted).",
                self.provider, self._accounts[self._current_idx].account_id,
            )
            return True

    def mark_rate_limited(
        self,
        key: str,
        retry_after: float,
        is_daily: bool = False,
        header_limit: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Tuple[bool, float, Optional[AccountKey]]:
        """
        Record rate limit / quota exhaustion on the specific account key.
        Then immediately finds another healthy account key to fail over to (preferring most headroom).

        Returns:
            (has_alternate_healthy_key: bool,
             min_wait_seconds: float,
             next_account: Optional[AccountKey])
        """
        with self._lock:
            now = time.time()
            account = self._key_map.get(key)
            if account:
                account.record_rate_limit(
                    retry_after=retry_after,
                    is_daily=is_daily,
                    header_limit=header_limit,
                    model=model,
                    now=now,
                )

            # Find alternate healthy account (excluding the failing key if possible)
            healthy = [acc for acc in self._accounts if acc.key != key and acc.is_available(now)]
            if healthy:
                # Alternate healthy account found! Pick the one with the most headroom / fewest tokens
                next_acc = min(
                    healthy,
                    key=lambda acc: (
                        acc.get_window_tokens(model=model, now=now),
                        acc.tokens_used or 0,
                        acc.last_used or 0.0,
                    ),
                )
                self._current_idx = next_acc.index
                logger.info(
                    "[%s] Immediate failover: switched to %s (healthy, least-loaded).",
                    self.provider, next_acc.account_id,
                )
                return True, 0.0, next_acc

            # No healthy accounts available right now. Calculate shortest wait time among all accounts
            waits = [acc.seconds_until_available(now) for acc in self._accounts if acc.status != KeyStatus.DISABLED]
            min_wait = min(waits) if waits else (retry_after or 15.0)
            return False, min_wait, None

    def mark_success(self, key: str, tokens: int = 0) -> None:
        with self._lock:
            account = self._key_map.get(key)
            if account:
                account.record_success(tokens=tokens or 0)

    def mark_error(self, key: str, error: Exception) -> None:
        with self._lock:
            account = self._key_map.get(key)
            if not account:
                return

            err_str = str(error).lower()
            is_auth = any(phrase in err_str for phrase in (
                "invalid api key", "unauthorized", "api_key_invalid", "authentication", "forbidden", "401", "403",
            ))
            account.record_error(is_auth_error=is_auth)
            if is_auth:
                now = time.time()
                healthy = [acc for acc in self._accounts if acc.is_available(now)]
                if healthy:
                    next_acc = min(
                        healthy,
                        key=lambda a: (a.get_window_tokens(now=now), a.tokens_used, a.last_used),
                    )
                    self._current_idx = next_acc.index

    def are_all_daily_exhausted(self) -> bool:
        """Returns True if every configured account has hit daily quota."""
        with self._lock:
            if not self._accounts:
                return False
            now = time.time()
            return all(
                acc.status in (KeyStatus.EXHAUSTED, KeyStatus.DISABLED) and not acc.is_available(now)
                for acc in self._accounts
            )

    def status_summary(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            accounts_data = [acc.to_dict() for acc in self._accounts]
            healthy_count = sum(1 for acc in self._accounts if acc.is_available(now))
            cooldown_count = sum(1 for acc in self._accounts if acc.status == KeyStatus.COOLDOWN and not acc.is_available(now))
            exhausted_count = sum(1 for acc in self._accounts if acc.status == KeyStatus.EXHAUSTED and not acc.is_available(now))
            disabled_count = sum(1 for acc in self._accounts if acc.status == KeyStatus.DISABLED)

            active_key = self.get_key()
            active_account = self._key_map.get(active_key)

            return {
                "provider": self.provider,
                "strategy": self.strategy,
                "total_accounts": len(self._accounts),
                "healthy_accounts": healthy_count,
                "cooldown_accounts": cooldown_count,
                "exhausted_accounts": exhausted_count,
                "disabled_accounts": disabled_count,
                "active_account_id": active_account.account_id if active_account else None,
                "active_index": self._current_idx if self._accounts else -1,
                "accounts": accounts_data,
            }
