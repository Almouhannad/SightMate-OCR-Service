from typing import List
import easyocr
import numpy as np
import cv2

from src.domain.models import OcrInput, OcrOutput, OcrResult, Rect
from src.domain.ports import OcrPort
from src.infrastructure.models.registry import register_adapter
from src.infrastructure.models.easyocr.config import easy_ocr_settings

@register_adapter("easyocr")
class EasyOCRAdapter(OcrPort):
    def __init__(self):
        # Initialize EasyOCR reader with settings from config
        self.reader = easyocr.Reader(
            lang_list=easy_ocr_settings.lang_list,
            gpu=easy_ocr_settings.gpu,
            model_storage_directory=easy_ocr_settings.model_storage_directory,
            user_network_directory=easy_ocr_settings.user_network_directory,
            detect_network=easy_ocr_settings.detect_network,
            recog_network=easy_ocr_settings.recog_network,
            download_enabled=easy_ocr_settings.download_enabled,
            detector=easy_ocr_settings.detector,
            recognizer=easy_ocr_settings.recognizer,
            verbose=easy_ocr_settings.verbose,
            quantize=easy_ocr_settings.quantize,
            cudnn_benchmark=easy_ocr_settings.cudnn_benchmark
        )

    def predict(self, data: OcrInput) -> OcrOutput:
        """Run OCR on the input image."""
        # Convert bytes to numpy array
        nparr = np.array(data.bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run OCR with settings from config
        result = self.reader.readtext(
            image=image,
            decoder=easy_ocr_settings.decoder,
            beamWidth=easy_ocr_settings.beamWidth,
            batch_size=easy_ocr_settings.batch_size,
            workers=easy_ocr_settings.workers,
            allowlist=easy_ocr_settings.allowlist,
            blocklist=easy_ocr_settings.blocklist,
            detail=easy_ocr_settings.detail,
            paragraph=easy_ocr_settings.paragraph,
            min_size=easy_ocr_settings.min_size,
            rotation_info=easy_ocr_settings.rotation_info,
            contrast_ths=easy_ocr_settings.contrast_ths,
            adjust_contrast=easy_ocr_settings.adjust_contrast,
            filter_ths=easy_ocr_settings.filter_ths,
            text_threshold=easy_ocr_settings.text_threshold,
            low_text=easy_ocr_settings.low_text,
            link_threshold=easy_ocr_settings.link_threshold,
            canvas_size=easy_ocr_settings.canvas_size,
            mag_ratio=easy_ocr_settings.mag_ratio,
            slope_ths=easy_ocr_settings.slope_ths,
            ycenter_ths=easy_ocr_settings.ycenter_ths,
            height_ths=easy_ocr_settings.height_ths,
            width_ths=easy_ocr_settings.width_ths,
            add_margin=easy_ocr_settings.add_margin,
            x_ths=easy_ocr_settings.x_ths,
            y_ths=easy_ocr_settings.y_ths,
            threshold=easy_ocr_settings.threshold,
            bbox_min_score=easy_ocr_settings.bbox_min_score,
            bbox_min_size=easy_ocr_settings.bbox_min_size,
            max_candidates=easy_ocr_settings.max_candidates,
            output_format=easy_ocr_settings.output_format
        )

        # Process results
        ocr_results = []
        for detection in result:
            bbox, text, confidence = detection[0], detection[1], detection[2]
            ocr_results.append(OcrResult(text=text, confidence=confidence, box=self.coords_to_rect(bbox)))

        return OcrOutput(texts=ocr_results) 
    
    def coords_to_rect(self, coords: List[List]) -> Rect:
        # Convert each numpy int into a Python float
        xs = [float(point[0]) for point in coords]
        ys = [float(point[1]) for point in coords]

        return Rect(
            left=min(xs),
            top=min(ys),
            right=max(xs),
            bottom=max(ys),
        )
    



