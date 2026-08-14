import numpy as np
import pytest

import lmms_eval.models.chat.internvl3 as internvl3_module
import lmms_eval.models.chat.internvl3_streaming as streaming_module
from lmms_eval.api.instance import Instance
from lmms_eval.models.chat.internvl3 import InternVL3, get_index
from lmms_eval.models.chat.internvl3_streaming import StreamingInternVL3


def test_max_fps_is_applied_before_frame_cap():
    indices = get_index(None, fps=30, max_frame=299, num_segments=512, max_fps=1)

    assert len(indices) == 10
    assert np.all(np.diff(indices) > 0)


def test_frame_count_is_capped_for_long_videos():
    indices = get_index(None, fps=30, max_frame=30 * 3600 - 1, num_segments=512, max_fps=1)

    assert len(indices) == 512


def test_short_video_still_yields_one_frame():
    indices = get_index(None, fps=30, max_frame=14, num_segments=512, max_fps=1)

    assert len(indices) == 1


@pytest.mark.parametrize("fps,max_frames,max_fps", [(0, 32, 1), (30, 0, 1), (30, 32, 0)])
def test_invalid_sampling_arguments_fail(fps, max_frames, max_fps):
    with pytest.raises(ValueError):
        get_index(None, fps=fps, max_frame=299, num_segments=max_frames, max_fps=max_fps)


@pytest.mark.parametrize(
    ("model_cls", "model_module"),
    [
        (InternVL3, internvl3_module),
        (StreamingInternVL3, streaming_module),
    ],
)
def test_generate_until_uses_standard_instance_fields(monkeypatch, model_cls, model_module):
    model = object.__new__(model_cls)
    model._rank = 0
    model._generate_until_with_prefetch = lambda *_args, **_kwargs: (
        ["A"],
        ["raw response"],
        1.0,
        1,
    )
    request = Instance(
        request_type="generate_until",
        arguments=(),
        idx=0,
        metadata={"task": "test", "doc_id": 0, "repeats": 1},
    )
    monkeypatch.setattr(model_module, "log_metrics", lambda **_kwargs: None)

    assert model.generate_until([request]) == ["A"]
    assert not hasattr(request, "raw_resps")
