import dataclasses
import logging
import multiprocessing as mp
import queue
import time

import tyro

from main import Args as EvalArgs
from main import eval_libero
from openpi_client.local_process_policy import LocalProcessClientPolicy
from openpi.serving.local_policy_worker import run_policy_worker


@dataclasses.dataclass
class PolicyWorkerArgs:
    # Must match your OpenPI training config name, e.g. "pi05_libero"
    config_name: str = "pi05_libero"

    # Path or gs:// path to checkpoint
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"

    # Optional prompt fallback
    default_prompt: str | None = None

    # For pytorch models only; e.g. "cpu", "cuda", "cuda:0"
    pytorch_device: str | None = None

    # Queue sizes
    request_queue_size: int = 8
    response_queue_size: int = 8

    # Timeout for client waiting policy response
    client_timeout_s: float = 300.0


@dataclasses.dataclass
class Args:
    eval: EvalArgs = dataclasses.field(default_factory=EvalArgs)
    policy: PolicyWorkerArgs = dataclasses.field(default_factory=PolicyWorkerArgs)


def _safe_put_close_signal(request_queue) -> None:
    try:
        request_queue.put_nowait({"type": "close"})
    except queue.Full:
        # Fallback to blocking put for cleanup
        request_queue.put({"type": "close"})


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][MAIN][%(levelname)s] %(message)s",
        force=True,
    )

    # Very important for JAX / PyTorch / mujoco-ish stacks
    mp.set_start_method("spawn", force=True)

    logging.info("creating IPC queues")
    request_queue = mp.Queue(maxsize=args.policy.request_queue_size)
    response_queue = mp.Queue(maxsize=args.policy.response_queue_size)

    logging.info("starting policy worker process")
    worker = mp.Process(
        target=run_policy_worker,
        kwargs={
            "request_queue": request_queue,
            "response_queue": response_queue,
            "config_name": args.policy.config_name,
            "checkpoint_dir": args.policy.checkpoint_dir,
            "default_prompt": args.policy.default_prompt,
            "pytorch_device": args.policy.pytorch_device,
            "worker_name": "policy_worker_0",
        },
        daemon=False,
    )
    worker.start()

    logging.info("policy worker started with pid=%s", worker.pid)

    client = LocalProcessClientPolicy(
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_s=args.policy.client_timeout_s,
        name="libero_eval_client",
    )

    try:
        eval_libero(args.eval, client=client)
    except KeyboardInterrupt:
        logging.info("main interrupted by keyboard interrupt")
    finally:
        logging.info("shutting down client/worker")
        try:
            client.close()
        except Exception:
            logging.exception("error while closing client")

        try:
            _safe_put_close_signal(request_queue)
        except Exception:
            logging.exception("failed to send close signal to worker")

        worker.join(timeout=20.0)

        if worker.is_alive():
            logging.warning("worker did not exit in time, terminating")
            worker.terminate()
            worker.join(timeout=5.0)

        logging.info("shutdown complete")
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)