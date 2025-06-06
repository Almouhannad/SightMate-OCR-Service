import io
from typing import Tuple
import numpy as np
from PIL import Image
import onnxruntime as ort

from src.domain.models import OcrInput, OcrOutput, OcrResult, Rect
from src.domain.ports import OcrPort
from src.infrastructure.models.paddleocr.postprocessing import post_process
from src.infrastructure.models.paddleocr.preprocessing import (
    preprocess_for_det,
    preprocess_recognize,
)
from src.infrastructure.models.registry import register_adapter
from src.infrastructure.models.paddleocr.config import paddle_ocr_settings


@register_adapter("paddleocr")
class PaddleOCRAdapter(OcrPort):
    def __init__(self):
        # Load ONNX models
        self.det_sess = ort.InferenceSession(
            paddle_ocr_settings.det_model_path,
            providers=paddle_ocr_settings.providers,
        )
        self.rec_sess = ort.InferenceSession(
            paddle_ocr_settings.rec_model_path,
            providers=paddle_ocr_settings.providers,
        )
        # Load character dictionary
        with open(paddle_ocr_settings.char_dict_path, encoding="utf8") as f:
            self.chars = [line.rstrip("\n") for line in f]

    def ctc_decode(self, pred: np.ndarray) -> Tuple[str, float]:
        """
        Decode CTC output to text and estimate confidence as the mean of max probablities
        """
        # pred shape: [1, T, C]
        logits = pred.squeeze(0)  # shape [T, C]
        # Softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)  # shape [T, C]
        max_probs = probs.max(axis=1)  # shape [T]
        confidence = float(max_probs.mean())

        idxs = logits.argmax(axis=1)
        blank = len(self.chars) - 1
        text_chars = []
        prev = None
        for i in idxs:
            if i != prev and i != blank:
                text_chars.append(self.chars[i])
            prev = i
        text = "".join(text_chars)
        return text, confidence

    def predict(self, data: OcrInput) -> OcrOutput:
        """Run OCR on the input bytes, return list of OcrResult"""
        # Load original image and note dimensions
        original_image = Image.open(io.BytesIO(bytes(data.bytes)))
        orig_w, orig_h = original_image.size

        # Preprocess for detection
        det_tensor, resized_pil = preprocess_for_det(bytes(data.bytes))
        resized_w, resized_h = resized_pil.size

        # Run text detection
        det_name = self.det_sess.get_inputs()[0].name
        det_out = self.det_sess.run(
            [self.det_sess.get_outputs()[0].name],
            {det_name: det_tensor},
        )[0].squeeze(0).squeeze(0)

        # Post-process: get quadrilateral boxes (list of 4-point coords) and crops
        boxes, crops = post_process(det_out, resized_pil)

        results = []
        for quad, crop in zip(boxes, crops):
            # Scale quadrilateral points back to original image size
            scaled_points = [
                [
                    float(pt[0] * orig_w / resized_w),
                    float(pt[1] * orig_h / resized_h),
                ]
                for pt in quad
            ]
            # Convert to LTRB
            xs = [p[0] for p in scaled_points]
            ys = [p[1] for p in scaled_points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)

            # Preprocess for recognition
            rec_tensor = preprocess_recognize(crop)

            # Run text recognition
            rec_name = self.rec_sess.get_inputs()[0].name
            pred = self.rec_sess.run(
                [self.rec_sess.get_outputs()[0].name],
                {rec_name: rec_tensor},
            )[0]  # shape [1, T, C]

            # Decode CTC output
            text, confidence = self.ctc_decode(pred)

            results.append(
                OcrResult(
                    text=text,
                    confidence=confidence,
                    box=Rect(
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    ),
                )
            )

        return OcrOutput(texts=results)
