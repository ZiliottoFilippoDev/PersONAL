from scipy.special import expit as sigmoid
from time import time
from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Dict, Any

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from vlfm.vlm.detections import ObjectDetections


class Owlv2_Detector_t:

    def __init__(self, 
                 detect_thresh: float = 0.4, 
                 model_id: str = "google/owlv2-base-patch16-ensemble", # "google/owlv2-base-patch16",
                 precision: str = "auto",
                 device=None,
                    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

        # Pick precision
        if self.device == "cuda":
            if precision == "fp16" or (precision == "auto" and torch.cuda.get_device_capability()[0] >= 7):
                self.autocast_dtype = torch.float16
                self.model.to(dtype=torch.float16)
            elif precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
                self.autocast_dtype = torch.bfloat16
                self.model.to(dtype=torch.bfloat16)
            else:
                self.autocast_dtype = None
        else:
            self.autocast_dtype = None

        self.query_txt = None
        self.detect_thresh = detect_thresh

    def set_query(self, texts):

        if isinstance(texts, str):
            self.query_txt = [[texts]]
        elif isinstance(texts, list):
            self.query_txt = [texts]
        else:
            raise TypeError("texts must be a str or List[str]")

        print(f"Set text query.")

    @torch.inference_mode()
    def is_query_in_image(self, target_image: torch.Tensor, plot_result: bool=False):

        if self.query_txt is None:
            raise RuntimeError("Call set_query(...) first.")

        target_image = self._ensure_pil(target_image)

        inputs = self.processor(text=self.query_txt, images=target_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        #Target image sizes to resclae box preds
        target_sizes = torch.tensor([(target_image.height, target_image.width)], device=self.device)

        #Convert outputs (bbox, logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.detect_thresh,
        )

        result = results[0]
        boxes, scores, text_labels = result["boxes"], result["scores"], result["labels"]

        # Normalize (x1, y1, x2, y2) by width and height
        h, w = target_image.height, target_image.width
        norm_factors = torch.tensor([w, h, w, h], device=boxes.device)
        boxes = boxes / norm_factors

        if len(boxes > 0):
            print(f"Top Detection Score: {max(scores)}")
    
        else:
            print(f"No Detections found under threshold ({self.detect_thresh})")

        if plot_result:
            self.plot_image_with_bbox(target_image, boxes.detach().cpu().numpy(), scores.detach().cpu().numpy())
    
        return scores, boxes

    @torch.inference_mode()
    def predict(self, image: np.ndarray):

        scores, boxes = self.is_query_in_image(image)

        detection = ObjectDetections(
            boxes = boxes.detach().cpu(),
            logits = scores.detach().cpu(),
            phrases = self.query_txt,
            image_source = image,
            fmt = "xyxy"
        )
        return detection


    # ----- Utils ------


    @staticmethod
    def _ensure_pil(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:

        #If PIL
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        #If Numpy Array
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return Image.fromarray(img.astype(np.uint8)).convert("RGB")

        #If Torch Tensor
        if isinstance(img, torch.Tensor):
            # Accept CHW [0..1] or [0..255], or HWC
            t = img.detach().cpu()
            if t.ndim == 3:
                if t.shape[0] in (1,3):  # CHW
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255)
                    t = t.permute(1,2,0).numpy()
                else:  # HWC
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255).numpy()
            else:
                raise ValueError("Expected 3D tensor image (CHW or HWC).")
            return Image.fromarray(t).convert("RGB")
        raise TypeError("Unsupported image type. Use PIL.Image, numpy array, or torch.Tensor.")

    def plot_image_with_bbox(self, image_pil, boxes, similarities):

        assert len(boxes) == len(similarities), "Please provide similarity values corresponding to the boxes. Should be equal length lists."

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_axis_off()

        ax.imshow(image_pil)

        # Normalize (x1, y1, x2, y2) by width and height
        h, w = image_pil.height, image_pil.width
        norm_factors = np.array([w, h, w, h])

        for sim, box_xyxy in zip(similarities, boxes):


            box_xyxy = box_xyxy * norm_factors

            x1, y1, x2, y2 = [float(v) for v in box_xyxy]

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, linewidth=2))

            # add similarity score text above the box
            ax.text(
                x1, y1 - 5, f"{sim:.3f}",
                fontsize=10, color="white", backgroundcolor="red"
            )


        ax.set_title(f'Top {len(boxes)} objects by Similarity')
        plt.show()



### TODO: Hosting the detector on a local server
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image
from typing import Optional


class Owlv2_Client:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/owlvit"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections



if __name__ == "__main__":

    #TODO: Hosting on a server
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading OwLViT Detector model...")

    class OwLv2_Server(ServerMixin, Owlv2_Detector_t):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    owlvit = OwLv2_Server()

    print("Owlv2 Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owlvit, name="owlvit", port=args.port)