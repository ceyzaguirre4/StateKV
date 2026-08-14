"""Mixin to add video prefetching capabilities to any model."""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Tuple, Any, Dict
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages


class VideoPrefetchMixin:
    """
    Mixin that adds video prefetching to any model's generate_until.

    Models that inherit from this mixin can use _generate_until_with_prefetch
    to automatically prefetch video data for the next batch while processing
    the current batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefetch_executor: ThreadPoolExecutor | None = None

    def _init_prefetch_executor(self) -> None:
        """Initialize the thread pool executor for prefetching (lazy init)."""
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="video_prefetch"
            )

    def _prepare_chat_messages_from_chunk(
        self, chunk: Tuple
    ) -> Tuple[Tuple, List[Dict], List[ChatMessages]]:
        """
        Extract and prepare chat messages from a chunk.

        Args:
            chunk: Tuple from collator containing (ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split)

        Returns:
            Tuple of (original_chunk, chat_messages_raw, chat_messages)
        """
        ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
        chat_messages_raw = [
            doc_to_messages[idx](self.task_dict[task][split][ids])
            for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
        ]
        chat_messages = [
            ChatMessages(**{"messages": message}) for message in chat_messages_raw
        ]
        return chunk, chat_messages_raw, chat_messages

    def _generate_until_with_prefetch(
        self,
        requests: List[Instance],
        load_vision_fn: Callable[[List[Dict], List[ChatMessages]], Any],
        process_chunk_fn: Callable[[Tuple, Any], Tuple[List[str], List[str], Dict]],
    ) -> Tuple[List[str], List[str], float, int]:
        """
        Generic prefetching loop that works with any video loading strategy.

        This method implements the common pattern of:
        1. Load vision data (prefetched from previous iteration)
        2. Start loading next batch's vision data in background
        3. Run inference on current batch (GPU busy while next loads)
        4. Process results

        Args:
            requests: List of instances to process
            load_vision_fn: Function that loads vision data
                Takes (chat_messages_raw, chat_messages) and returns vision_data
            process_chunk_fn: Function that processes one chunk
                Takes (chunk_data, vision_data) and returns (results, raw_responses, metrics)
                where metrics is a dict with "latency" and "tokens" keys

        Returns:
            Tuple of (results, raw_responses, total_latency, total_tokens)
        """
        self._init_prefetch_executor()

        res: List[str] = []
        raw_resps: List[str] = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(
            total=num_iters, disable=(self.rank != 0), desc="Model Responding"
        )

        e2e_latency = 0.0
        total_tokens = 0
        prefetch_future = None

        for i, chunk in enumerate(chunks):
            # Get vision data (from prefetch or load synchronously)
            if prefetch_future is not None:
                chunk_data, vision_data = prefetch_future.result()
            else:
                # First iteration - load synchronously
                chunk_data = self._prepare_chat_messages_from_chunk(chunk)
                _, chat_messages_raw, chat_messages = chunk_data
                vision_data = load_vision_fn(chat_messages_raw, chat_messages)

            # Start prefetching next batch
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]

                def _prefetch_next():
                    next_chunk_data = self._prepare_chat_messages_from_chunk(
                        next_chunk
                    )
                    _, next_chat_messages_raw, next_chat_messages = next_chunk_data
                    next_vision_data = load_vision_fn(
                        next_chat_messages_raw, next_chat_messages
                    )
                    return next_chunk_data, next_vision_data

                prefetch_future = self._prefetch_executor.submit(_prefetch_next)
            else:
                prefetch_future = None

            # Process this chunk (inference happens here while next batch loads)
            chunk_res, chunk_raw, metrics = process_chunk_fn(chunk_data, vision_data)
            res.extend(chunk_res)
            raw_resps.extend(chunk_raw)

            e2e_latency += metrics.get("latency", 0.0)
            total_tokens += metrics.get("tokens", 0)

            pbar.update(1)

        pbar.close()

        # Reorder results back to original order
        res = re_ords.get_original(res)
        raw_resps = re_ords.get_original(raw_resps)

        return res, raw_resps, e2e_latency, total_tokens
