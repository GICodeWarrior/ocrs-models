from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import resize, InterpolationMode

from .hiertext import DEFAULT_ALPHABET
from .util import SizedDataset, encode_text, transform_image


@dataclass(frozen=True)
class SynthFontConfig:
    ttf_path: str
    output_height: int = 64
    background_gray: int = 144
    foreground_white: int = 255
    min_width: int = 10
    max_width: int = 800

    # Render text at these nominal font pixel heights before resizing to
    # `output_height` for the OCRS recognition model.
    render_heights: tuple[int, ...] = (12, 16, 32)

    # JPEG compression quality used for the augmented variants (0-95).
    jpeg_quality: int = 89


class SyntheticFontRecognition(SizedDataset):
    """
    Synthetic dataset for the text recognition model.

    Generates every 1-, 2-, and 3-character string over DEFAULT_ALPHABET,
    and all 4-character uppercase strings (A-Z only, no spaces), and can
    optionally include an explicit short list of strings.

    Space rule for generated strings:
      - No spaces in 1-char or 2-char strings
      - For 3-char strings, spaces are allowed ONLY in the middle
      - For 4-char uppercase strings, only A-Z are used

    Each (render_height, item) pair produces TWO samples:
      1. A clean render.
      2. The same render passed through JPEG compression at `jpeg_quality`.

    Output sample format matches HierTextRecognition:
      {
        "image": CHW float tensor in [-0.5, 0.5], height == 64,
        "text_seq": [seq] int tensor of class indices (CTC blank reserved at 0),
      }
    """

    def __init__(
        self,
        config: SynthFontConfig,
        train: bool = True,
        transform=None,
        max_images: Optional[int] = None,
        alphabet: Optional[list[str]] = None,
        explicit_strings: Optional[list[str]] = None,
    ):
        super().__init__()
        self.cfg = config
        self.transform = transform

        if alphabet is None:
            alphabet = [c for c in DEFAULT_ALPHABET]
        self.alphabet = alphabet
        self._chars = alphabet
        self._space = " "

        if not self.cfg.render_heights:
            raise ValueError("render_heights must not be empty")
        for h in self.cfg.render_heights:
            if h <= 0:
                raise ValueError(f"render_heights must be positive, got {h}")

        # Restrict 4-char expansion to uppercase ASCII letters that actually
        # exist in the configured alphabet.
        self._upper_chars = [c for c in self._chars if "A" <= c <= "Z"]

        if explicit_strings is None:
            explicit_strings = [
                # Crate Storage
                "Aircraft Depot",
                "Seaport",
                "Storage Depot",
                # Item Storage
                "Border Base",
                "Bunker Base",
                "Encampment",
                "Keep",
                "Relic Base",
                "Safe House",
                "Town Base",
                # Common stockpile name
                "Public",
            ]
        self.explicit_strings = self._validate_explicit_strings(explicit_strings)

        # Mixed index:
        #   generated items: ("gen", (...))
        #   explicit items:  ("str", text)
        base_index = self._build_index()

        # Expand each item for every render height, with a clean and a
        # JPEG-compressed variant (quality `jpeg_quality`) for each combination.
        self._index: list[tuple[int, tuple[str, object], bool]] = []
        for render_height in self.cfg.render_heights:
            for item in base_index:
                self._index.append((render_height, item, False))   # clean
                self._index.append((render_height, item, True))    # jpeg q89

        if max_images is not None:
            self._index = self._index[:max_images]

        self.train = train

    def _validate_explicit_strings(self, strings: list[str]) -> list[str]:
        allowed = set(self.alphabet)
        out: list[str] = []

        for s in strings:
            if not s:
                continue
            bad = [ch for ch in s if ch not in allowed]
            if bad:
                raise ValueError(
                    f"explicit string {s!r} contains chars not in alphabet: {bad}"
                )
            out.append(s)

        return out

    def _build_index(self) -> list[tuple[str, object]]:
        """
        Mixed index entries:
          - ("gen", (1, i, -1))
          - ("gen", (2, i, j))
          - ("gen", (3, i, j*len + k))
          - ("gen_upper4", packed)
          - ("str", "hello")

        Rules for generated entries:
          - No spaces in 1-char or 2-char strings
          - For 3-char strings, space is allowed ONLY in the middle
          - 4-char strings are uppercase A-Z only
        """
        n = len(self._chars)
        idx: list[tuple[str, object]] = []

        space = self._space

        # Append explicit strings
        for s in self.explicit_strings:
            idx.append(("str", s))

        # 1-char (no space)
        for i in range(n):
            if self._chars[i] == space:
                continue
            idx.append(("gen", (1, i, -1)))

        # 2-char (no spaces anywhere)
        for i in range(n):
            if self._chars[i] == space:
                continue
            for j in range(n):
                if self._chars[j] == space:
                    continue
                idx.append(("gen", (2, i, j)))

        # 3-char: space allowed ONLY in the middle
        for i in range(n):
            if self._chars[i] == space:
                continue
            for j in range(n):
                for k in range(n):
                    if self._chars[k] == space:
                        continue
                    idx.append(("gen", (3, i, j * n + k)))

        # 4-char uppercase only: A-Z, no spaces
        m = len(self._upper_chars)
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    for d in range(m):
                        packed = ((a * m + b) * m + c) * m + d
                        idx.append(("gen_upper4", packed))

        return idx

    def __len__(self) -> int:
        return len(self._index)

    @lru_cache(maxsize=32)
    def _load_font(self, pixel_height: int) -> ImageFont.FreeTypeFont:
        # For PIL FreeType fonts, `size` is the nominal pixel size.
        return ImageFont.truetype(self.cfg.ttf_path, size=pixel_height)

    def _decode_generated_item_to_text(self, spec: tuple[int, int, int]) -> str:
        n = len(self._chars)
        length, a, bc = spec
        if length == 1:
            return self._chars[a]
        if length == 2:
            return self._chars[a] + self._chars[bc]
        b = bc // n
        c = bc % n
        return self._chars[a] + self._chars[b] + self._chars[c]

    def _decode_upper4_item_to_text(self, packed: int) -> str:
        m = len(self._upper_chars)
        a = packed // (m * m * m)
        rem = packed % (m * m * m)
        b = rem // (m * m)
        rem = rem % (m * m)
        c = rem // m
        d = rem % m
        return (
            self._upper_chars[a]
            + self._upper_chars[b]
            + self._upper_chars[c]
            + self._upper_chars[d]
        )

    def _decode_item_to_text(self, item: tuple[str, object]) -> str:
        kind, payload = item
        if kind == "gen":
            return self._decode_generated_item_to_text(payload)  # type: ignore[arg-type]
        if kind == "gen_upper4":
            return self._decode_upper4_item_to_text(payload)  # type: ignore[arg-type]
        if kind == "str":
            return payload  # type: ignore[return-value]
        raise ValueError(f"unknown index kind: {kind!r}")

    def _render_text_line(self, text: str, render_height: int) -> torch.Tensor:
        font = self._load_font(render_height)

        # First measure the text tightly.
        tmp = Image.new("L", (1, 1), color=self.cfg.background_gray)
        d = ImageDraw.Draw(tmp)
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)

        text_w = max(1, right - left)
        text_h = max(1, bottom - top)

        # Small safety margin around all sides to avoid clipping/aliasing spill.
        margin = max(1, int(round(render_height / 8)))

        canvas_w = text_w + 2 * margin
        canvas_h = text_h + 2 * margin

        canvas_w = max(1, canvas_w)
        canvas_h = max(1, canvas_h)

        img = Image.new(
            "L",
            (canvas_w, canvas_h),
            color=self.cfg.background_gray,
        )
        draw = ImageDraw.Draw(img)

        # Offset by bbox origin so the drawn glyphs fit tightly in the canvas.
        x = margin - left
        y = margin - top
        draw.text((x, y), text, font=font, fill=self.cfg.foreground_white)

        arr = np.array(img, dtype=np.uint8)
        chw = torch.from_numpy(arr).unsqueeze(0)
        return chw

    @staticmethod
    def _apply_jpeg_compression(img_chw: torch.Tensor, quality: int) -> torch.Tensor:
        """Round-trip a CHW uint8 tensor through JPEG at the given quality.

        The tensor must be single-channel (grayscale) uint8.  The result is
        returned in the same shape and dtype.
        """
        arr = img_chw.squeeze(0).numpy()          # HW uint8
        pil_img = Image.fromarray(arr, mode="L")

        buf = io.BytesIO()
        # JPEG doesn't support mode "L" on all Pillow builds; convert to RGB,
        # compress, then convert back to ensure compatibility.
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_img_back = Image.open(buf).convert("L")

        arr_back = np.array(pil_img_back, dtype=np.uint8)
        return torch.from_numpy(arr_back).unsqueeze(0)

    def _resize_to_output_height(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        output_height = self.cfg.output_height
        aspect = w / max(1, h)
        output_width = int(round(output_height * aspect))
        output_width = max(self.cfg.min_width, min(self.cfg.max_width, output_width))

        img = resize(
            img,
            [output_height, output_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return img.clamp(-0.5, 0.5)

    def __getitem__(self, idx: int) -> dict:
        render_height, item, use_jpeg = self._index[idx]
        text = self._decode_item_to_text(item)

        img_u8 = self._render_text_line(text, render_height)

        if use_jpeg:
            img_u8 = self._apply_jpeg_compression(img_u8, self.cfg.jpeg_quality)

        img = transform_image(img_u8)

        if self.transform is not None:
            img = self.transform(img)
            img = img.clamp(-0.5, 0.5)

        # Always normalize to the OCRS model's fixed input height.
        img = self._resize_to_output_height(img)
        img = img.contiguous().clone()

        text_seq = encode_text(text, self.alphabet, unknown_char="?")
        text_seq = text_seq.contiguous()

        return {"image": img, "text_seq": text_seq}

