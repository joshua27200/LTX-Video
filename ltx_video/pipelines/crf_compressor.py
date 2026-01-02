import av
import torch
import io
import numpy as np


def _encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def _decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def compress(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    original_height, original_width = image.shape[0], image.shape[1]
    # H.264 requires even dimensions, so crop to even if needed
    cropped_height = (original_height // 2) * 2
    cropped_width = (original_width // 2) * 2

    image_array = (image[:cropped_height, :cropped_width] * 255.0).byte().cpu().numpy()
    with io.BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = _decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0

    # Pad back to original dimensions if needed
    if tensor.shape[0] != original_height or tensor.shape[1] != original_width:
        result = torch.zeros(
            original_height,
            original_width,
            tensor.shape[2],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        result[:cropped_height, :cropped_width] = tensor
        # Copy edge pixels to fill the padding
        if original_height > cropped_height:
            result[cropped_height:, :cropped_width] = tensor[-1:, :cropped_width]
        if original_width > cropped_width:
            result[:cropped_height, cropped_width:] = tensor[:cropped_height, -1:]
        if original_height > cropped_height and original_width > cropped_width:
            result[cropped_height:, cropped_width:] = tensor[-1, -1]
        tensor = result

    return tensor
