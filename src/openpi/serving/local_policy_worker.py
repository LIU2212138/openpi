import logging
import os
import queue
import time
import traceback
from typing import Any

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)


def build_policy_from_config(
    *,
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a trained policy inside the worker process."""
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    return policy


def run_policy_worker(
    request_queue,
    response_queue,
    *,
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    worker_name: str = "policy_worker",
    poll_timeout_s: float = 1.0,
) -> None:
    """Main loop for the policy worker process.

    Important:
        - model should be loaded inside the child process
        - evaluator and worker communicate only through queues
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][{worker_name}][%(levelname)s] %(message)s",
        force=True,
    )

    logger.info("starting worker pid=%s", os.getpid())
    logger.info("loading policy config=%s checkpoint_dir=%s", config_name, checkpoint_dir)

    try:
        policy = build_policy_from_config(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        logger.info("policy loaded successfully")
    except Exception:
        logger.exception("failed to load policy")
        raise

    while True:
        try:
            try:
                message = request_queue.get(timeout=poll_timeout_s)
            except queue.Empty:
                continue

            msg_type = message.get("type")

            if msg_type == "close":
                logger.info("received close signal, exiting worker loop")
                break

            if msg_type != "infer":
                logger.warning("received unknown message type: %s", msg_type)
                continue

            request_id = message["request_id"]
            obs = message["obs"]

            logger.debug("processing infer request_id=%s", request_id)
            start_time = time.monotonic()

            try:
                result = policy.infer(obs)
                elapsed_ms = (time.monotonic() - start_time) * 1000.0

                # Add worker-side timing without clobbering policy_timing
                if isinstance(result, dict):
                    result = dict(result)
                    result["worker_timing"] = {
                        "roundtrip_in_worker_ms": elapsed_ms,
                    }

                response_queue.put(
                    {
                        "type": "result",
                        "request_id": request_id,
                        "result": result,
                    }
                )
            except Exception:
                tb = traceback.format_exc()
                logger.error("error while handling request_id=%s\n%s", request_id, tb)
                response_queue.put(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": tb,
                    }
                )

        except KeyboardInterrupt:
            logger.info("worker interrupted by keyboard interrupt")
            break
        except Exception:
            logger.exception("unexpected fatal exception in worker loop")
            raise

    logger.info("worker stopped")