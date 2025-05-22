import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import io
from src.domain.ports import OCRPort
from src.domain.models import OCRInput, OCROutput, TextBlock
from src.infrastructure.models.registry import register_adapter
from src.infrastructure.models.paddleocr.config import paddle_ocr_settings
from src.infrastructure.models.paddleocr.preprocessing import preprocess_for_det, preprocess_recognize
from src.infrastructure.models.paddleocr.postprocessing import post_process
from src.infrastructure.models.paddleocr.helpers import order_points

@register_adapter("paddleocr")
class PaddleOCRAdapter(OCRPort):
    def __init__(self):
        # Load ONNX models
        self.det_sess = ort.InferenceSession(
            paddle_ocr_settings.det_model_path,
            providers=paddle_ocr_settings.providers
        )
        self.rec_sess = ort.InferenceSession(
            paddle_ocr_settings.rec_model_path,
            providers=paddle_ocr_settings.providers
        )
        # Load character dictionary
        with open(paddle_ocr_settings.char_dict_path, encoding="utf8") as f:
            self.chars = [line.rstrip("\n") for line in f]

    def ctc_decode(self, pred: np.ndarray) -> str:
        """Decode CTC output to text."""
        idxs = pred.argmax(axis=2).squeeze(0)
        blank = len(self.chars)-1
        txt, prev = [], None
        for i in idxs:
            if i != prev and i != blank:
                txt.append(self.chars[i])
            prev = i
        return "".join(txt)

    def _create_annotated_image(self, image_bytes: bytes, boxes: list) -> bytes:
        """Create an annotated image with text regions drawn."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        image = image.resize(paddle_ocr_settings.target_size, Image.BILINEAR)

        draw = ImageDraw.Draw(image)
        for box in boxes:
            # Order points for consistent drawing
            ordered_box = order_points(box)
            # Convert to tuples for PIL drawing and add first point at end to close polygon
            pts = [tuple(pt) for pt in ordered_box] + [tuple(ordered_box[0])]
            draw.line(pts, fill="black", width=4)

        # Resize back to original dimensions
        image = image.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def predict(self, data: OCRInput) -> OCROutput:
        """Run OCR on the input image."""
        # Preprocess for detection
        det_tensor, resized_pil = preprocess_for_det(data.image_bytes)
        
        # Run detection
        det_name = self.det_sess.get_inputs()[0].name
        det_map = self.det_sess.run(
            [self.det_sess.get_outputs()[0].name],
            {det_name: det_tensor}
        )[0].squeeze(0).squeeze(0)

        # Post-process detection results
        boxes, crops = post_process(det_map, resized_pil)

        # Recognize text in each crop
        blocks = []
        for box, crop in zip(boxes, crops):
            # Preprocess for recognition
            rec_tensor = preprocess_recognize(crop)
            
            # Run recognition
            rec_name = self.rec_sess.get_inputs()[0].name
            pred = self.rec_sess.run(
                [self.rec_sess.get_outputs()[0].name],
                {rec_name: rec_tensor}
            )[0]
            
            # Decode text
            text = self.ctc_decode(pred)
            blocks.append(TextBlock(text=text))

        # Create annotated image
        annotated_image = self._create_annotated_image(data.image_bytes, boxes)

        return OCROutput(blocks=blocks, annotated_image=annotated_image)
