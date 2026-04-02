import itertools
import logging
import queue
import time
from typing import Any


logger = logging.getLogger(__name__)


class LocalProcessClientPolicy:
    """A local multiprocessing client that mimics the websocket client interface.

    It sends inference requests to a policy worker process through queues.
    """

    def __init__(
        self,
        request_queue,
        response_queue,
        *,
        timeout_s: float = 300.0,
        name: str = "local_policy_client",
    ):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._timeout_s = timeout_s
        self._name = name
        self._request_counter = itertools.count()
        self._pending_responses: dict[int, dict[str, Any]] = {}

    def infer(self, obs: dict) -> dict:
        request_id = next(self._request_counter)
        request_message = {
            "type": "infer",
            "request_id": request_id,
            "obs": obs,
            "timestamp": time.time(),
        }

        logger.debug("[%s] sending infer request_id=%s", self._name, request_id)
        self._request_queue.put(request_message)

        return self._wait_for_response(request_id)

    def close(self) -> None:
        """Optional client-side close hook."""
        logger.info("[%s] client closed", self._name)

    def _wait_for_response(self, request_id: int) -> dict:
        if request_id in self._pending_responses:
            message = self._pending_responses.pop(request_id)
            return self._handle_response_message(message, request_id)

        deadline = time.monotonic() + self._timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"[{self._name}] timed out waiting for response to request_id={request_id} "
                    f"after {self._timeout_s} seconds"
                )

            try:
                message = self._response_queue.get(timeout=remaining)
            except queue.Empty as e:
                raise TimeoutError(
                    f"[{self._name}] timed out waiting for response to request_id={request_id} "
                    f"after {self._timeout_s} seconds"
                ) from e

            msg_request_id = message.get("request_id")
            msg_type = message.get("type")

            if msg_type in ("result", "error") and msg_request_id == request_id:
                return self._handle_response_message(message, request_id)

            # If later you add concurrency / out-of-order responses, this cache becomes useful.
            if msg_request_id is not None:
                self._pending_responses[msg_request_id] = message
            else:
                logger.warning("[%s] received message without request_id: %s", self._name, message)

    def _handle_response_message(self, message: dict, request_id: int) -> dict:
        msg_type = message.get("type")

        if msg_type == "result":
            result = message["result"]
            logger.debug("[%s] received result for request_id=%s", self._name, request_id)
            return result

        if msg_type == "error":
            error_text = message.get("error", "unknown worker error")
            raise RuntimeError(
                f"[{self._name}] policy worker returned error for request_id={request_id}:\n{error_text}"
            )

        raise RuntimeError(
            f"[{self._name}] unexpected response type for request_id={request_id}: {msg_type}, message={message}"
        )