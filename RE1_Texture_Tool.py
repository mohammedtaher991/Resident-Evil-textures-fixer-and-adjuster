#!/usr/bin/env python3
"""
RE1 Texture Tool - replace the character texture inside a Resident Evil 1 (PS1) .EMD model
with a PNG (recommended) or a .TIM, keeping the game's transparency (shadow) intact.

Needs: Python 3 + Pillow + numpy   (pip install pillow numpy)
"""
import os
import shutil
import struct
import sys

import numpy as np
from PIL import Image

# =====================================================================================
#  CORE (no GUI) - TIM / EMD handling
# =====================================================================================


def parse_tim(d, o):
    """Parse a 4-bit or 8-bit CLUT TIM located at offset o of bytes d (the kinds RE1 model textures use)."""
    if o + 20 > len(d) or struct.unpack_from('<I', d, o)[0] != 0x10:
        raise ValueError("Not a TIM")
    flag = struct.unpack_from('<I', d, o + 4)[0]
    if flag not in (8, 9):
        raise ValueError("Only 4 bit and 8 bit CLUT TIMs are supported here")
    bpp = 4 if flag == 8 else 8
    cb, cx, cy, cw, ch = struct.unpack_from('<IHHHH', d, o + 8)
    if cw != (16 if bpp == 4 else 256) or ch < 1 or cb != 12 + cw * ch * 2:
        raise ValueError("Unexpected CLUT size")
    clut = np.frombuffer(d, '<u2', cw * ch, o + 20).reshape(ch, cw).copy()
    p = o + 8 + cb
    pb, px, py, pw, ph = struct.unpack_from('<IHHHH', d, p)
    if pb != 12 + pw * 2 * ph or p + pb > len(d):
        raise ValueError("Unexpected pixel block size")
    raw = np.frombuffer(d, np.uint8, pw * 2 * ph, p + 12).reshape(ph, pw * 2)
    if bpp == 8:
        pix = raw.copy()
    else:                                    # 4 bit: two pixels per byte, the LOW nibble is the left pixel
        pix = np.stack([raw & 15, raw >> 4], axis=-1).reshape(ph, pw * 4).astype(np.uint8)
    return dict(start=o, end=p + pb, clut=clut, cx=cx, cy=cy, pix=pix, px=px, py=py,
                w=pix.shape[1], h=ph, bpp=bpp)


def find_tim(d):
    """Find the (last) valid 4-bit or 8-bit CLUT TIM inside an EMD file."""
    best, i = None, 0
    while True:
        i = d.find(b'\x10\x00\x00\x00', i)
        if i < 0:
            break
        try:
            best = parse_tim(d, i)
        except Exception:
            pass
        i += 1
    if best is None:
        raise ValueError("No 4 bit or 8 bit CLUT TIM texture found in this file.")
    return best


def section_padding(emd, tim_end):
    """
    In an RE1 EMD the texture section is the TIM followed by zero padding (8 bytes in EM1033) and then the footer.
    RE123MV Editor writes/expects the TIM WITH that padding; a TIM without it makes the texture section too short
    and the game can crash. Returns the number of padding bytes to append (default 8).
    """
    if emd is None:
        return 8
    tail = emd[tim_end:]
    return min(len(tail) - len(tail.lstrip(b'\0')), 8) or 8


def same_file(a, b):
    """True if the two paths point at the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def backup_path(path):
    """path.bak, or path.bak2, path.bak3 ... - an older backup is never overwritten."""
    p, n = path + '.bak', 2
    while os.path.exists(p):
        p = '%s.bak%d' % (path, n)
        n += 1
    return p


def write_file_safely(path, data, backup=True):
    """
    Write data to path. If the file already exists and backup is on, it is copied to a .bak first.
    The new data goes to a temporary file that is checked and then moved into place, so a failed save
    can never leave a half-written EMD. Returns the backup path (or None).
    """
    bak = None
    if backup and os.path.exists(path):
        bak = backup_path(path)
        shutil.copy2(path, bak)
    tmp = path + '.tmp_re1'
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        with open(tmp, 'rb') as f:
            if f.read() != data:
                raise IOError("The written file did not match - nothing was changed.")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return bak


def build_tim(clut, pix, cx, cy, px, py, bpp=8):
    """CLUT TIM (4 bit: 16 colours per palette row, 8 bit: 256). pix holds palette indices."""
    rows, cols = clut.shape
    h, w = pix.shape
    if cols != (16 if bpp == 4 else 256):
        raise ValueError("A %d bit CLUT needs %d colours per palette row." % (bpp, 16 if bpp == 4 else 256))
    if bpp == 4:
        if w % 4:
            raise ValueError("4 bit textures need a width that is a multiple of 4.")
        if int(pix.max()) > 15:
            raise ValueError("4 bit pixels can only use palette entries 0-15.")
        p = pix.astype(np.uint8)
        data, pw = (p[:, 0::2] | (p[:, 1::2] << 4)).astype(np.uint8), w // 4   # low nibble = left pixel
    else:
        data, pw = pix.astype(np.uint8), w // 2
    out = struct.pack('<II', 0x10, 8 if bpp == 4 else 9)
    out += struct.pack('<IHHHH', 12 + rows * cols * 2, cx, cy, cols, rows) + clut.astype('<u2').tobytes()
    out += struct.pack('<IHHHH', 12 + data.size, px, py, pw, h) + data.tobytes()
    return out


def build_direct_tim(data, bpp, px, py):
    """
    Direct-colour TIM, no palette.
      16 bit: data is (h, w) uint16, each pixel = 5 bit R,G,B + the transparency (STP) bit.
      24 bit: data is (h, w, 3) uint8 RGB. There is no transparency bit in this mode.
    """
    if bpp == 16:
        h, w = data.shape
        body, pw, flag = data.astype('<u2').tobytes(), w, 2
    else:
        h, w = data.shape[:2]
        if (w * 3) % 2:
            raise ValueError("24 bit textures need an even width.")
        body, pw, flag = data.astype(np.uint8).tobytes(), w * 3 // 2, 3     # width is counted in 16 bit words
    return struct.pack('<II', 0x10, flag) + struct.pack('<IHHHH', 12 + len(body), px, py, pw, h) + body


def vals_to_rgb(v):
    v = v.astype(np.int32)
    return np.stack([(v & 31) << 3, ((v >> 5) & 31) << 3, ((v >> 10) & 31) << 3], axis=-1)


def row_map(rows, w, h):
    """The sheet is split evenly across the CLUT rows (left half = palette 0, right half = palette 1)."""
    return (np.arange(w) * rows // w)[None, :].repeat(h, axis=0)


def tim_to_rgba(d):
    """Read any TIM (4/8/16/24 bit) as an RGBA image. Pure 0x0000 pixels become transparent."""
    o = d.find(b'\x10\x00\x00\x00')
    if o < 0:
        raise ValueError("This doesn't look like a TIM file.")
    flag = struct.unpack_from('<I', d, o + 4)[0]
    mode, has_clut = flag & 3, flag & 8
    p, clut = o + 8, None
    if has_clut:
        cb, cx, cy, cw, ch = struct.unpack_from('<IHHHH', d, p)
        clut = np.frombuffer(d, '<u2', cw * ch, p + 12).reshape(ch, cw)
        p += cb
    pb, px, py, pw, ph = struct.unpack_from('<IHHHH', d, p)
    raw = p + 12
    if mode in (0, 1):
        if mode == 0:
            b = np.frombuffer(d, np.uint8, pw * 2 * ph, raw).reshape(ph, pw * 2)
            idx = np.stack([b & 15, b >> 4], axis=-1).reshape(ph, pw * 4)
        else:
            idx = np.frombuffer(d, np.uint8, pw * 2 * ph, raw).reshape(ph, pw * 2)
        rows = clut.shape[0]
        vals = clut[row_map(rows, idx.shape[1], ph), idx]
    elif mode == 2:
        vals = np.frombuffer(d, '<u2', pw * ph, raw).reshape(ph, pw)
    else:
        wd = pw * 2 // 3
        rgb = np.frombuffer(d, np.uint8, wd * 3 * ph, raw).reshape(ph, wd, 3)
        return np.dstack([rgb, np.full((ph, wd), 255, np.uint8)])
    rgb = vals_to_rgb(vals).astype(np.uint8)
    rgb[vals == 0x8000] = 8                  # opaque black stays OPAQUE (a pure black pixel would be read as see-through)
    return np.dstack([rgb, np.where(vals == 0, 0, 255).astype(np.uint8)])


def texture_to_rgba(t):
    """The texture a model currently has, as an RGBA image (pure 0x0000 pixels are transparent)."""
    v = t['clut'][row_map(t['clut'].shape[0], t['w'], t['h']), t['pix']]
    rgb = vals_to_rgb(v).astype(np.uint8)
    rgb[v == 0x8000] = 8                     # opaque black (0x8000) -> near black, so it is not turned see-through
    return np.dstack([rgb, np.where(v == 0, 0, 255).astype(np.uint8)])


def load_source(path):
    """Load the replacement image: PNG/JPG/BMP/... or .TIM -> RGBA uint8 array."""
    if path.lower().endswith('.tim'):
        return tim_to_rgba(open(path, 'rb').read())
    return np.array(Image.open(path).convert('RGBA'))


def texture_health(t):
    """
    Check the transparency (STP) flags of a texture.
      'ok'   : only a small part of the pixels is semi-transparent (= the shadow, like the original game file)
      'none' : no semi-transparent pixels at all (no shadow)
      'all'  : most pixels are semi-transparent - another tool set the flag on everything (damaged)
    """
    v = t['clut'][row_map(t['clut'].shape[0], t['w'], t['h']), t['pix']]
    stp = ((v & 0x8000) != 0) & (v != 0x8000)
    n, frac = int(stp.sum()), float(stp.mean())
    # Untouched game files flag the shadow and some DARK colours (e.g. Barry: a third of the sheet, all dark cloth).
    # A tool that flagged everything also flags the bright colours (skin, faces), so judge by the bright pixels.
    lum = vals_to_rgb(v) @ np.array([0.3, 0.59, 0.11])
    bright = (lum > 100) & (v != 0)
    bright_share = float((stp & bright).sum()) / max(1, int(bright.sum()))
    state = 'none' if n == 0 else 'all' if (frac > 0.60 or bright_share > 0.30) else 'ok'
    return dict(state=state, pixels=n, percent=100.0 * frac, bright_percent=100.0 * bright_share)


def stp_pixel_count(t):
    """Number of pixels that use semi-transparent (STP) palette entries = the shadow texture."""
    return texture_health(t)['pixels']


PLATFORM_NOTE = ("Note: on the PC version this will probably look fine, but on the PS1 version "
                 "the shadows will show broken colours.")


def damaged_message(h):
    if h['state'] == 'none':
        what = ("This texture has NO semi-transparent (shadow) pixels, but the original game texture has them "
                "(a small part of the sheet). Keeping this palette means a missing or flat shadow and strange floor colours.")
    else:
        what = ("The transparency flags look damaged: %.0f%% of this texture's pixels are semi-transparent, but in the "
                "original game file only the shadow and some dark colours are. Keeping this palette causes strange floor colours "
                "and wrong shadows." % h['percent'])
    return (what + "\n\n" + PLATFORM_NOTE + "\n\nFix: put the untouched original EM file in 'Original game .EMD' "
            "(recommended), or use 'Rebuild palette'.")


SHADOW_WARNING = ("This texture has NO semi-transparent (shadow) pixels.\n\n"
                  "The original game texture has them. If they are missing, your model's transparency was probably "
                  "lost by an earlier save, and 'Keep game palette' would keep that damage (strange floor colours, "
                  "flat or missing shadows).\n\n" + PLATFORM_NOTE + "\n\n"
                  "Fix: put the untouched original EM file in 'Original game .EMD', or use 'Rebuild palette'.")


def palette_fit(ref, src):
    """How well can the reference (original game) palettes show this image?  Returns (mean error, share of pixels
    that are far off).  Errors are colour distances on the 0-441 scale.  Same artwork / same colours -> near 0;
    a different outfit with new colours -> large."""
    clut = ref['clut']
    rows, h, w = clut.shape[0], ref['h'], ref['w']
    rm = row_map(rows, w, h)
    ref_vals = clut[rm, ref['pix']]
    rgb = (src[..., :3].astype(np.int32) >> 3) << 3
    use = (src[..., 3] >= 128) & ((ref_vals & 0x8000) == 0) & (ref_vals != 0)
    errs = []
    for r in range(rows):
        pal = vals_to_rgb(clut[r][(clut[r] & 0x8000) == 0])
        m = use & (rm == r)
        if not m.any() or not len(pal):
            continue
        cols, inv = np.unique(rgb[m].reshape(-1, 3), axis=0, return_inverse=True)
        d = np.sqrt(((cols[:, None, :] - pal[None, :, :]) ** 2).sum(-1)).min(1)
        errs.append(d[inv.reshape(-1)])
    if not errs:
        return 0.0, 0.0
    e = np.concatenate(errs)
    return float(e.mean()), float((e > 32).mean())


def same_pixel_share(ref, src, tol=32):
    """Share of (opaque, non-shadow) pixels whose colour is within tol of the reference pixel at the same place.
    The same artwork -> ~1.0 (even when lightly retouched); a different outfit / layout -> about a third."""
    clut = ref['clut']
    rm = row_map(clut.shape[0], ref['w'], ref['h'])
    v = clut[rm, ref['pix']]
    use = (src[..., 3] >= 128) & ((v & 0x8000) == 0) & (v != 0)
    if not use.any():
        return 1.0
    rgb = (src[..., :3].astype(np.int32) >> 3) << 3
    d = np.sqrt(((vals_to_rgb(v) - rgb) ** 2).sum(-1))
    return float((d[use] <= tol).mean())


HOLE_MIN_COMPONENTS = 10   # this many tiny transparent specks inside the art = the "invisible letters" kind of original


def find_holes(ref):
    """Tiny transparent specks (1-6 pixels) inside the artwork of the reference, e.g. the letters of the STARS badge
    in the original Director's Cut file, which are see-through there.  Returns (mask, number of specks)."""
    rm = row_map(ref['clut'].shape[0], ref['w'], ref['h'])
    tr = (ref['clut'][rm, ref['pix']] == 0)
    h, w = tr.shape
    seen = np.zeros((h, w), bool)
    mask = np.zeros((h, w), bool)
    n = 0
    for y0, x0 in zip(*np.nonzero(tr)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        todo, pts = [(y0, x0)], []
        while todo:
            y, x = todo.pop()
            pts.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and tr[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        if len(pts) <= 6:
            ys, xs = zip(*pts)
            mask[ys, xs] = True
            n += 1
    return mask, n


def has_holes(ref):
    return find_holes(ref)[1] >= HOLE_MIN_COMPONENTS


LAYOUT_MATCH = 0.75      # edge correlation needed for the image to count as the same layout as the reference


def layout_similarity(ref, src):
    """How alike the picture STRUCTURE is (where the edges and shapes are), ignoring colours: ~1.0 for the same
    layout even when it is recoloured, ~0.2 or less for another character / outfit sheet."""
    clut = ref['clut']
    rm = row_map(clut.shape[0], ref['w'], ref['h'])
    v = clut[rm, ref['pix']]
    use = (src[..., 3] >= 128) & ((v & 0x8000) == 0) & (v != 0)
    if use.sum() < 200:
        return 1.0                                     # too little to judge: trust the reference

    def edges(rgb):
        lum = rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
        return (gx + gy)[use]
    a = edges(vals_to_rgb(v).astype(np.float64))
    b = edges(((src[..., :3].astype(np.int32) >> 3) << 3).astype(np.float64))
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 1.0


def shadow_blob_mask(stp):
    """The round ground-shadow blob among the semi-transparent pixels (mask of the same shape).
    Other semi-transparent areas (strips and margins around a model's own clothes panels) are not round
    and are ignored, so a reference with a different layout can't paste its margins into the new texture."""
    spot = np.zeros_like(stp)
    spot[199:231, 80:109] = True                          # where every RE1 character sheet has its ground shadow
    found = _round_blobs(stp & spot, 4000)
    return found if found.any() else _round_blobs(stp, 900)


def _round_blobs(stp, max_area):
    h, w = stp.shape
    seen = np.zeros((h, w), bool)
    out = np.zeros((h, w), bool)
    for y0, x0 in zip(*np.nonzero(stp)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        todo, pts = [(y0, x0)], []
        while todo:
            y, x = todo.pop()
            pts.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and stp[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        ys, xs = zip(*pts)
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        fill = len(pts) / float(bw * bh)
        if 100 <= len(pts) <= max_area and 0.5 <= bw / float(bh) <= 2.0 and 0.70 <= fill <= 0.90:      # an ellipse fills ~0.785
            out[ys, xs] = True
    return out


def suggest_mode(ref, src):
    """'keep' when the original palettes fit the image (same-looking texture: colours get cleaned up to the game's
    own, lost letters come back); 'rebuild' when they don't (different outfit: keep its colours, only protect the shadow)."""
    mean, bad = palette_fit(ref, src)
    same = same_pixel_share(ref, src)
    # same-looking texture = most pixels are (nearly) where the original has them AND the palettes can show the rest
    return 'keep' if (same >= 0.70 and bad < 0.08 and mean < 12) else 'rebuild'


DEFAULT_OPTS = dict(mode='keep',                    # 'keep' = keep game palette | 'rebuild' = like the Photoshop plugin
                    except_black_translucent=False,  # rebuild mode only
                    black_transparent=True,          # rebuild mode only
                    keep_shadow=True,                # rebuild mode only (always on in 'keep' mode)
                    honor_alpha=False,               # treat transparent pixels of the new image as transparent
                    bpp=None,                        # output mode: 4 / 8 (CLUT) or 16 / 24 (Direct); None = same as the model
                    source_name='your image')        # only used in messages


def convert(ref, src, opts):
    """
    ref  : parsed TIM whose palettes / transparency bits are the reference (the original game texture)
    src  : RGBA uint8 array (h, w, 4) with the new artwork
    Returns dict(pix, clut, vals, notes, changed)
    """
    clut_ref, ref_pix = ref['clut'], ref['pix']
    rows, h, w = clut_ref.shape[0], ref['h'], ref['w']
    target = opts.get('bpp') or ref['bpp']             # 4 / 8 = CLUT modes, 16 / 24 = Direct modes
    ncol_t = 16 if target == 4 else 256
    depth_change = target in (4, 8) and target != ref['bpp']
    if src.shape[:2] != (h, w):
        raise ValueError("The new image must be %dx%d pixels (it is %dx%d)." % (w, h, src.shape[1], src.shape[0]))
    rm = row_map(rows, w, h)
    ref_vals = clut_ref[rm, ref_pix]
    ref_rgb = vals_to_rgb(ref_vals)
    ref_stp = (ref_vals & 0x8000) != 0
    notes = []
    health = texture_health(ref)
    # A reference that does not look like the new texture has a different layout: its pixel positions mean nothing here.
    # Then only the round shadow blob is carried over, and transparent areas of the new image stay transparent.
    mismatch = bool(ref_stp.any()) and layout_similarity(ref, src) < LAYOUT_MATCH
    ignore_holes = bool(opts.get('ignore_holes')) and has_holes(ref)
    if mismatch and target in (4, 8) and opts['mode'] == 'keep':
        opts = dict(opts, mode='rebuild')       # another character's palettes can't show this texture: build new ones
    ref_stp_all = ref_stp                  # every semi-transparent pixel of the reference
    if mismatch:
        ref_stp = shadow_blob_mask(ref_stp)
    if target in (4, 8) and opts['mode'] == 'keep':
        if depth_change:
            raise ValueError("Changing between 4 bit and 8 bit CLUT needs 'Rebuild palette' "
                             "(the game's palettes can't be kept at a different colour depth).")
        if health['state'] != 'ok':
            raise ValueError(damaged_message(health))

    rgb = (src[..., :3].astype(np.int32) >> 3) << 3         # PS1 keeps 5 bits per channel
    transp = src[..., 3] < 128
    if ignore_holes:
        notes.append("The reference has see-through specks inside the artwork (like invisible STARS letters): those "
                     "pixels were ignored and left as they are in your texture; everything else follows the reference.")
    if mismatch:
        notes.append("The reference texture has a different layout, so only its ground shadow was carried over.")
    if opts['honor_alpha'] or mismatch:
        rgb = np.where(transp[..., None], 0, rgb)
    else:
        kept = transp & (ref_vals != 0)
        rgb = np.where(transp[..., None], ref_rgb, rgb)
        if kept.any():
            notes.append("%d transparent pixels in %s were opaque in the original game texture; "
                         "the original pixel was kept." % (int(kept.sum()), opts.get('source_name', 'your image')))

    if target == 24:
        # no palette and no transparency bit: keep all 8 bits of every colour from the source image
        full = np.where(transp[..., None], 0 if (opts['honor_alpha'] or mismatch) else ref_rgb, src[..., :3].astype(np.int32)).astype(np.uint8)
        rgb5 = (full[..., 0].astype(np.int32) >> 3) | ((full[..., 1].astype(np.int32) >> 3) << 5) | \
               ((full[..., 2].astype(np.int32) >> 3) << 10)
        vals = np.where(rgb5 == 0, 0x8000, rgb5).astype(np.uint16)          # preview only
        notes.append("24 bit Direct: full 8 bit colours, no palette and no transparency bit.")
        return dict(pix=None, clut=None, rgb24=full, vals=vals, bpp=24, notes=notes, changed=int((vals != ref_vals).sum()))
    if target == 16:
        # no palette: every pixel has its own 5 bit colour + the transparency bit, set by the two plugin options
        rgb5 = (rgb[..., 0] >> 3) | ((rgb[..., 1] >> 3) << 5) | ((rgb[..., 2] >> 3) << 10)
        stp_bit = 0x8000 if opts['except_black_translucent'] else 0
        vals = np.where(rgb5 == 0, 0 if opts['black_transparent'] else 0x8000, rgb5 | stp_bit)
        if opts['keep_shadow'] and health['state'] == 'ok':
            vals = np.where(ref_stp, ref_vals, vals)       # shadow pixels keep their original colour + flag
            notes.append("16 bit Direct: no palette limit; shadow pixels from the original kept.")
        else:
            notes.append("16 bit Direct: no palette limit; shadow pixels are NOT preserved.")
        vals = vals.astype(np.uint16)
        return dict(pix=None, clut=None, vals=vals, bpp=16, notes=notes, changed=int((vals != ref_vals).sum()))

    new_pix = ref_pix.copy()
    new_clut = np.zeros((rows, ncol_t), np.uint16) if depth_change else clut_ref.copy()

    if opts['mode'] == 'keep':
        todo = (~np.all(rgb == ref_rgb, axis=-1) | ref_stp_all) & ~ref_stp   # semi-transparent but not shadow: re-pick
        for r in range(rows):
            sel = todo & (rm == r)
            if not sel.any():
                continue
            safe = np.where((clut_ref[r] & 0x8000) == 0)[0]
            sc = vals_to_rgb(clut_ref[r][safe])
            u, inv = np.unique(rgb[sel], axis=0, return_inverse=True)
            d2 = ((u[:, None, :] - sc[None, :, :]) ** 2).sum(-1)
            nonblack = u.any(axis=1)
            d2[np.ix_(nonblack, clut_ref[r][safe] == 0)] = 10 ** 9   # never turn a coloured pixel transparent
            new_pix[sel] = safe[d2.argmin(1)][inv.reshape(-1)]
        notes.append("Palettes and transparency bits kept from the original game texture.")
    else:
        # only trust the reference's shadow if its flags are healthy (a damaged reference would "protect" everything)
        prot = ref_stp if (opts['keep_shadow'] and health['state'] == 'ok' and not depth_change) else np.zeros_like(ref_stp)
        stp_bit = 1 if opts['except_black_translucent'] else 0
        for r in range(rows):
            cm = rm == r
            pm = prot & cm
            reserved = sorted(set(ref_pix[pm].tolist()))
            avail = [i for i in range(ncol_t) if i not in set(reserved)]
            sel = cm & ~pm
            row = np.zeros(ncol_t, np.uint16)
            for i in reserved:
                row[i] = clut_ref[r][i]
            colors = rgb[sel]
            idx_map = np.zeros(len(colors), np.int64)
            if len(colors):
                isb = ~colors.any(axis=1)
                if isb.any():
                    bi = avail.pop(0)
                    row[bi] = 0 if opts['black_transparent'] else 0x8000
                    idx_map[isb] = bi
                nb = colors[~isb]
                if len(nb):
                    K = max(1, len(avail))
                    q = Image.fromarray(nb.astype(np.uint8).reshape(-1, 1, 3)).quantize(
                        colors=K, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                    qi = np.array(q).reshape(-1)
                    pal = (np.array(q.getpalette()[:K * 3]).reshape(-1, 3).astype(np.int32) >> 3) << 3
                    for j, c in enumerate(pal):
                        if not c.any():
                            c = np.array([0, 0, 8])           # never let a real colour become pure black
                        row[avail[j]] = (c[0] >> 3) | ((c[1] >> 3) << 5) | ((c[2] >> 3) << 10) | (stp_bit << 15)
                    idx_map[~isb] = np.array(avail)[qi]
            new_pix[sel] = idx_map
            new_clut[r] = row
        if depth_change:
            notes.append("Colour depth changed to %d bit CLUT: palettes rebuilt from your image (at most %d colours per palette); "
                         "the shadow transparency can NOT be carried over." % (target, ncol_t))
        else:
            notes.append("Palettes rebuilt from your image (%s)." %
                         ("shadow pixels from the original kept" if opts['keep_shadow'] and health['state'] == 'ok'
                          else "the reference has no usable shadow, so the shadow is NOT preserved - add the original EMD to keep it"
                          if opts['keep_shadow'] else "shadow NOT preserved"))

    vals = new_clut[rm, new_pix]
    return dict(pix=new_pix, clut=new_clut, vals=vals, bpp=target, notes=notes, changed=int((vals != ref_vals).sum()))


def render_rgba(vals, show_translucent=False):
    """Preview image: transparent pixels -> checkerboard, optional pink tint on semi-transparent (STP) pixels."""
    h, w = vals.shape
    rgb = vals_to_rgb(vals).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 8 + xx // 8) % 2)[..., None] == 0, 200, 150).astype(np.float32).repeat(3, -1)
    out = np.where((vals == 0)[..., None], checker, rgb)
    if show_translucent:
        m = ((vals & 0x8000) != 0) & (vals != 0x8000)
        out[m] = out[m] * 0.45 + np.array([255, 0, 200], np.float32) * 0.55
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), 'RGB')


# =====================================================================================
#  GUI
# =====================================================================================

def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    class App:
        def __init__(self, root):
            self.root = root
            root.title("RE1 Texture Tool")
            self.model_path, self.ref_path, self.src_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
            self.mode = tk.StringVar(value='keep')
            self.ebt = tk.BooleanVar(value=False)
            self.bt = tk.BooleanVar(value=True)
            self.keep_shadow = tk.BooleanVar(value=True)
            self.honor_alpha = tk.BooleanVar(value=False)
            self.cx, self.cy, self.px, self.py = (tk.IntVar(value=v) for v in (0, 480, 0, 0))
            self.zoom = tk.IntVar(value=200)              # preview zoom in percent (100 = actual size)
            self.zoom_text = tk.StringVar(value='200')    # what is typed in the zoom box
            self.sizes = {}                               # shown image size per preview tab (pixels)
            self.last_depth = 8
            self.ignore_holes = False                    # reference with invisible-letter specks: use only its shadow
            self.hole_answers = {}                       # reference path -> user's answer (asked once per file)
            self.depth = tk.IntVar(value=8)               # image data mode: 4 / 8 = CLUT, 16 / 24 = Direct
            self.overlay = tk.BooleanVar(value=False)
            self.status = tk.StringVar(value="Step 1: choose your model (.EMD).")
            self.model_bytes = self.ref_bytes = None
            self.model_t = self.ref_t = self.src = self.result = None
            self.no_shadow = False
            self.pad = 8
            self.backup = tk.BooleanVar(value=True)
            self.replace = tk.BooleanVar(value=True)      # False = keep the model's own texture
            self.src_widgets = ()
            self.health = None
            self.last_error = ''
            self.pal_note = tk.StringVar(value='')
            self.photos = {}
            self.build()

        # ---------- layout ----------
        def build(self):
            # The status line stays fixed at the bottom; everything above it scrolls when the window is small.
            status_bar = ttk.Label(self.root, textvariable=self.status, relief='sunken', anchor='w', wraplength=1000)
            status_bar.pack(fill='x', side='bottom', padx=8, pady=6)
            status_bar.bind('<Configure>', lambda e: status_bar.configure(wraplength=max(200, e.width - 20)))
            r =self.make_scroll_area()
            top = ttk.LabelFrame(r, text=" Files ")
            top.pack(fill='x', padx=8, pady=6)
            rows = [("Your model (.EMD):", self.model_path, self.pick_model),
                    ("Original game .EMD (recommended):", self.ref_path, self.pick_ref),
                    ("New texture (PNG recommended, or .TIM):", self.src_path, self.pick_src)]
            for i, (label, var, cmd) in enumerate(rows):
                gr = i if i < 2 else i + 1                # grid row 2 holds the 'replace texture' tick box
                ttk.Label(top, text=label).grid(row=gr, column=0, sticky='w', padx=6, pady=2)
                entry = ttk.Entry(top, textvariable=var, width=70)
                entry.grid(row=gr, column=1, padx=4)
                btn = ttk.Button(top, text="Browse...", command=cmd)
                btn.grid(row=gr, column=2, padx=4)
                if i == 2:
                    self.src_widgets = (entry, btn)
            ttk.Checkbutton(top, text="Replace the texture with a new image   (untick = keep my model's current texture and "
                            "only fix its palettes / transparency)", variable=self.replace,
                            command=self.on_replace).grid(row=2, column=0, columnspan=3, sticky='w', padx=6)
            ttk.Label(top, foreground='gray', text="The original file supplies the palettes and shadow. Strongly recommended if your model "
                      "was saved with other tools; if left empty, your model's own palettes are used.").grid(row=4, column=0, columnspan=3, sticky='w', padx=6)
            self.pal_label = ttk.Label(top, textvariable=self.pal_note, wraplength=900, justify='left')
            self.pal_label.grid(row=5, column=0, columnspan=3, sticky='w', padx=6, pady=(2, 0))

            mid = ttk.Frame(r)
            mid.pack(fill='both', expand=True, padx=8)
            left = ttk.Frame(mid)
            left.pack(side='left', fill='both', expand=True)
            bar = ttk.Frame(left)
            bar.pack(fill='x')
            ttk.Label(bar, text="Zoom:").pack(side='left')
            ttk.Button(bar, text="-", width=3, command=lambda: self.step_zoom(-1)).pack(side='left')
            self.zoom_entry = ttk.Entry(bar, textvariable=self.zoom_text, width=6, justify='center')
            self.zoom_entry.pack(side='left', padx=2)
            self.zoom_entry.bind('<Return>', self.apply_zoom_text)
            self.zoom_entry.bind('<FocusOut>', self.apply_zoom_text)
            ttk.Label(bar, text="%").pack(side='left')
            ttk.Button(bar, text="+", width=3, command=lambda: self.step_zoom(1)).pack(side='left', padx=(2, 0))
            ttk.Button(bar, text="Fit", width=4, command=self.fit_zoom).pack(side='left', padx=(4, 0))
            ttk.Label(bar, foreground='gray', text="(mouse wheel zooms, drag to move)").pack(side='left', padx=(6, 0))
            ttk.Checkbutton(bar, text="Highlight semi-transparent (shadow) areas", variable=self.overlay,
                            command=self.refresh).pack(side='left', padx=10)
            self.nb = ttk.Notebook(left)
            self.nb.pack(fill='both', expand=True)
            self.views = {}
            self.view_keys = []
            for key, title in (('now', "In your model now"), ('new', "New result"), ('ref', "Original (reference)")):
                f = ttk.Frame(self.nb)
                self.nb.add(f, text=title)
                f.rowconfigure(0, weight=1)
                f.columnconfigure(0, weight=1)
                cv = tk.Canvas(f, highlightthickness=0, width=520, height=420)
                vb = ttk.Scrollbar(f, orient='vertical', command=cv.yview)
                hb = ttk.Scrollbar(f, orient='horizontal', command=cv.xview)
                cv.configure(yscrollcommand=vb.set, xscrollcommand=hb.set)
                cv.grid(row=0, column=0, sticky='nsew')
                vb.grid(row=0, column=1, sticky='ns')
                hb.grid(row=1, column=0, sticky='ew')
                item = cv.create_image(0, 0, anchor='nw')
                msg = cv.create_text(10, 10, anchor='nw', text="(nothing to show yet)")
                # mouse wheel = zoom (around the pointer); drag = move; scrollbars also work
                cv.bind('<MouseWheel>', lambda e, k=key: self.on_view_wheel(k, e))
                cv.bind('<Button-4>', lambda e, k=key: self.on_view_wheel(k, e, 1))
                cv.bind('<Button-5>', lambda e, k=key: self.on_view_wheel(k, e, -1))
                cv.bind('<ButtonPress-1>', lambda e, c=cv: c.scan_mark(e.x, e.y))
                cv.bind('<B1-Motion>', lambda e, c=cv: c.scan_dragto(e.x, e.y, gain=1))
                self.views[key] = (cv, item, msg)
                self.view_keys.append(key)

            right = ttk.LabelFrame(mid, text=" Settings ")
            right.pack(side='right', fill='y', padx=(8, 0))
            ttk.Label(right, text="Palette handling").pack(anchor='w', padx=6, pady=(4, 0))
            self.rb_keep = ttk.Radiobutton(right, text="Keep game palette\n(use with the original EMD - recommended)", value='keep',
                                           variable=self.mode, command=self.on_mode)
            self.rb_keep.pack(anchor='w', padx=14)
            self.rb_rebuild = ttk.Radiobutton(right, text="Rebuild palette (like the Photoshop plugin)\n(use when you have no original EMD)",
                                              value='rebuild', variable=self.mode, command=self.on_mode)
            self.rb_rebuild.pack(anchor='w', padx=14)
            ttk.Separator(right).pack(fill='x', pady=6)
            ttk.Label(right, text="Image data mode").pack(anchor='w', padx=6)
            self.rb_depth = {}
            for bits, label in ((4, "4 bit CLUT"), (8, "8 bit CLUT (the mode RE1 uses)"),
                                (16, "16 bit Direct"), (24, "24 bit Direct")):
                self.rb_depth[bits] = ttk.Radiobutton(right, text=label, value=bits, variable=self.depth,
                                                      command=self.on_depth)
                self.rb_depth[bits].pack(anchor='w', padx=14)
            self.depth_note = ttk.Label(right, foreground='gray', wraplength=230, justify='left', text='')
            self.depth_note.pack(anchor='w', padx=14)
            ttk.Separator(right).pack(fill='x', pady=6)
            g = ttk.Frame(right)
            g.pack(padx=6)
            ttk.Label(g, text="CLUT section (VRAM)").grid(row=0, column=0, columnspan=4, sticky='w')
            ttk.Label(g, text="X").grid(row=1, column=0)
            self.clut_entries = (ttk.Entry(g, textvariable=self.cx, width=5), ttk.Entry(g, textvariable=self.cy, width=5))
            self.clut_entries[0].grid(row=1, column=1)
            ttk.Label(g, text="Y").grid(row=1, column=2)
            self.clut_entries[1].grid(row=1, column=3)
            self.clut_size = ttk.Label(g, text="Size: -")
            self.clut_size.grid(row=2, column=0, columnspan=4, sticky='w')
            ttk.Label(g, text="Pixel data section (VRAM)").grid(row=3, column=0, columnspan=4, sticky='w', pady=(6, 0))
            ttk.Label(g, text="X").grid(row=4, column=0)
            ttk.Entry(g, textvariable=self.px, width=5).grid(row=4, column=1)
            ttk.Label(g, text="Y").grid(row=4, column=2)
            ttk.Entry(g, textvariable=self.py, width=5).grid(row=4, column=3)
            self.pix_size = ttk.Label(g, text="Size: -")
            self.pix_size.grid(row=5, column=0, columnspan=4, sticky='w')
            ttk.Separator(right).pack(fill='x', pady=6)
            self.cb_ebt = ttk.Checkbutton(right, text="Except black -> Translucent", variable=self.ebt, command=self.update_preview)
            self.cb_bt = ttk.Checkbutton(right, text="Black -> Transparent", variable=self.bt, command=self.update_preview)
            self.cb_ks = ttk.Checkbutton(right, text="Keep shadow pixels from original", variable=self.keep_shadow, command=self.update_preview)
            for cb in (self.cb_ebt, self.cb_bt, self.cb_ks):
                cb.pack(anchor='w', padx=14)
            self.cb_ha = ttk.Checkbutton(right, text="Treat transparent pixels of my\nimage as transparent",
                                         variable=self.honor_alpha, command=self.update_preview)
            self.cb_ha.pack(anchor='w', padx=14, pady=(6, 0))
            ttk.Separator(right).pack(fill='x', pady=6)
            ttk.Button(right, text="Default", command=self.defaults).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save current texture as PNG...", command=self.export_png).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save over my modded .EMD", command=self.save_over).pack(fill='x', padx=10, pady=2)
            ttk.Checkbutton(right, text="Make a backup (.bak) first", variable=self.backup).pack(anchor='w', padx=22)
            ttk.Button(right, text="Save as new .EMD... (keeps mine)", command=self.save_emd).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save texture as .TIM...", command=self.save_tim).pack(fill='x', padx=10, pady=2)

            self.on_mode()

        def make_scroll_area(self):
            """A frame inside a canvas with vertical + horizontal scrollbars. Returns the inner frame."""
            outer = ttk.Frame(self.root)
            outer.pack(fill='both', expand=True)
            outer.rowconfigure(0, weight=1)
            outer.columnconfigure(0, weight=1)
            self.canvas = tk.Canvas(outer, highlightthickness=0)
            vbar = ttk.Scrollbar(outer, orient='vertical', command=self.canvas.yview)
            hbar = ttk.Scrollbar(outer, orient='horizontal', command=self.canvas.xview)
            self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
            self.canvas.grid(row=0, column=0, sticky='nsew')
            vbar.grid(row=0, column=1, sticky='ns')
            hbar.grid(row=1, column=0, sticky='ew')
            body = ttk.Frame(self.canvas)
            self.body = body
            self.body_id = self.canvas.create_window((0, 0), window=body, anchor='nw')
            body.bind('<Configure>', self.fit_scroll_area)
            self.canvas.bind('<Configure>', self.fit_scroll_area)
            # mouse wheel scrolls the page (Shift + wheel = sideways)
            self.root.bind_all('<MouseWheel>', self.on_page_wheel)
            self.root.bind_all('<Shift-MouseWheel>', lambda e: self.on_page_wheel(e, True))
            self.root.bind_all('<Button-4>', lambda e: self.on_page_wheel(e, False, -1))
            self.root.bind_all('<Button-5>', lambda e: self.on_page_wheel(e, False, 1))
            return body

        def fit_scroll_area(self, event=None):
            """Page is at least as big as the window (so the preview still grows), never smaller than its content."""
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            bw, bh = self.body.winfo_reqwidth(), self.body.winfo_reqheight()
            self.canvas.itemconfigure(self.body_id, width=max(cw, bw), height=max(ch, bh))
            self.canvas.configure(scrollregion=(0, 0, max(cw, bw), max(ch, bh)))

        def on_page_wheel(self, event, sideways=False, step=None):
            if step is None:
                step = -1 if event.delta > 0 else 1
            if sideways:
                self.canvas.xview_scroll(step * 3, 'units')
            else:
                self.canvas.yview_scroll(step * 3, 'units')

        # ---------- helpers ----------
        def native_bpp(self):
            t = self.ref_t or self.model_t
            return t['bpp'] if t else 8

        def update_sizes(self):
            t = self.ref_t or self.model_t
            if t is None:
                return
            d = self.depth.get()
            rows = t['clut'].shape[0]
            if d in (4, 8):
                self.clut_size.configure(text="Size: %d x %d" % (16 if d == 4 else 256, rows))
            else:
                self.clut_size.configure(text="(not used in Direct modes)")
            words = {4: t['w'] // 4, 8: t['w'] // 2, 16: t['w'], 24: t['w'] * 3 // 2}[d]
            self.pix_size.configure(text="Size: %d x %d (words: %d)" % (t['w'], t['h'], words))

        def on_depth(self):
            d, native = self.depth.get(), self.native_bpp()
            prev, self.last_depth = self.last_depth, d
            if d in (4, 8) and d != native:
                self.mode.set('rebuild')          # the original palettes can't be reused at another depth
            elif d == native and prev != d and self.health and self.health['state'] == 'ok':
                self.mode.set('keep')             # came back to the model's own depth
            self.on_mode()

        def on_mode(self):
            d, native = self.depth.get(), self.native_bpp()
            clut_mode = d in (4, 8)
            layout_ok = self.layout_ok()
            keep_ok = layout_ok
            if not keep_ok and self.mode.get() == 'keep':
                self.mode.set('rebuild')              # Keep can't work with another character's palettes / a speckled original
            self.rb_keep.configure(state='normal' if clut_mode and d == native and keep_ok else 'disabled')
            self.rb_rebuild.configure(state='normal' if clut_mode else 'disabled')
            rebuild = self.mode.get() == 'rebuild'
            keep_clut = clut_mode and self.mode.get() == 'keep'
            # Keep game palette = the original palettes, depth and shadow are kept as they are, so the other
            # depths and the 'transparent pixels' option can't apply; choose Rebuild palette to use them.
            for bits, rb in self.rb_depth.items():
                rb.configure(state='normal' if (not keep_clut or bits == native) else 'disabled')
            self.cb_ha.configure(state='disabled' if keep_clut else 'normal')
            if d == 24:
                states = ('disabled',) * 3                    # 24 bit has no transparency bit
            elif d == 16:
                states = ('normal', 'normal', 'disabled')     # shadow is always protected in 16 bit
            else:
                states = ('normal' if rebuild else 'disabled',) * 3
            for cb, st in zip((self.cb_ebt, self.cb_bt, self.cb_ks), states):
                cb.configure(state=st)
            for e in self.clut_entries:
                e.configure(state='normal' if clut_mode else 'disabled')
            notes = {4: "4 bit: at most 16 colours per palette.",
                     8: "",
                     16: "16 bit Direct: no palette, 5 bits per colour. Standalone .TIM export only.",
                     24: "24 bit Direct: full colour, no transparency. Standalone .TIM export only."}
            if d != native and clut_mode:
                notes[d] = "Different from the model's own depth: the game needs %d bit inside an EMD, so use this for .TIM export." % native
            elif d in (16, 24):
                notes[d] += " The game needs %d bit inside an EMD." % native
            self.depth_note.configure(text=notes[d])
            self.update_pal_note()
            self.update_sizes()
            self.update_preview()

        def defaults(self):
            self.depth.set(self.native_bpp())
            self.last_depth = self.native_bpp()
            # Keep game palette is only the safe default when the reference palette is not damaged
            self.mode.set('rebuild' if self.health and self.health['state'] != 'ok' else 'keep')
            self.ebt.set(False)
            self.bt.set(True)
            self.keep_shadow.set(True)
            self.honor_alpha.set(False)
            t = self.ref_t or self.model_t
            if t:
                self.cx.set(t['cx']); self.cy.set(t['cy']); self.px.set(t['px']); self.py.set(t['py'])
            self.on_mode()

        def opts(self):
            return dict(mode=self.mode.get(), except_black_translucent=self.ebt.get(), black_transparent=self.bt.get(),
                        keep_shadow=self.keep_shadow.get(),
                        honor_alpha=self.honor_alpha.get() and not (self.mode.get() == 'keep' and self.depth.get() in (4, 8)),
                        bpp=self.depth.get(), ignore_holes=self.ignore_holes,
                        source_name="your image" if self.replace.get() else "your model's texture")

        def show(self, key, vals):
            cv, item, msg = self.views[key]
            if vals is None:
                cv.itemconfigure(item, image='')
                cv.itemconfigure(msg, state='normal')
                cv.configure(scrollregion=(0, 0, 0, 0))
                self.sizes[key] = (0, 0)
                return
            img = render_rgba(vals, self.overlay.get())
            z = self.zoom.get() / 100.0
            w, h = max(1, round(img.width * z)), max(1, round(img.height * z))
            img = img.resize((w, h), Image.NEAREST)
            self.photos[key] = ImageTk.PhotoImage(img)
            cv.itemconfigure(item, image=self.photos[key])
            cv.itemconfigure(msg, state='hidden')
            cv.configure(scrollregion=(0, 0, w, h))
            self.sizes[key] = (w, h)

        ZOOM_STEPS = (10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1600, 3200)

        def set_zoom(self, pct, anchor=None):
            """anchor = (tab key, x, y) of the pointer: that spot of the image stays under the pointer."""
            pct = int(max(10, min(3200, round(pct))))
            fx = fy = None
            if anchor:
                key, x, y = anchor
                cv = self.views[key][0]
                ow, oh = self.sizes.get(key, (0, 0))
                if ow and oh:
                    fx, fy = cv.canvasx(x) / ow, cv.canvasy(y) / oh
            self.zoom.set(pct)
            self.zoom_text.set(str(pct))
            self.refresh()
            if fx is not None:
                nw, nh = self.sizes[key]
                cv.xview_moveto(max(0.0, (fx * nw - x) / nw))
                cv.yview_moveto(max(0.0, (fy * nh - y) / nh))

        def step_zoom(self, direction, anchor=None):
            cur = self.zoom.get()
            if direction > 0:
                nxt = next((s for s in self.ZOOM_STEPS if s > cur), self.ZOOM_STEPS[-1])
            else:
                nxt = next((s for s in reversed(self.ZOOM_STEPS) if s < cur), self.ZOOM_STEPS[0])
            self.set_zoom(nxt, anchor)

        def apply_zoom_text(self, event=None):
            try:
                v = float(self.zoom_text.get().strip().rstrip('%').strip())
                if v != v or v <= 0:
                    raise ValueError
            except ValueError:
                self.zoom_text.set(str(self.zoom.get()))      # not a number: show the real zoom again
                return
            self.set_zoom(v)

        def fit_zoom(self):
            t = self.ref_t or self.model_t
            iw, ih = (t['w'], t['h']) if t else (256, 256)
            cv = self.views[self.view_keys[self.nb.index(self.nb.select())]][0]
            w, h = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
            self.set_zoom(min(w / iw, h / ih) * 100 - 1)

        def on_view_wheel(self, key, event, step=None):
            if step is None:
                step = 1 if event.delta > 0 else -1
            self.step_zoom(step, (key, event.x, event.y))
            return 'break'                                    # don't also scroll the page

        def refresh(self):
            def vals_of(t):
                return None if t is None else t['clut'][row_map(t['clut'].shape[0], t['w'], t['h']), t['pix']]
            self.show('now', vals_of(self.model_t))
            self.show('ref', vals_of(self.ref_t or self.model_t))
            self.show('new', None if self.result is None else self.result['vals'])

        def fail(self, e):
            self.status.set("Error: %s" % e)
            messagebox.showerror("RE1 Texture Tool", str(e))

        # ---------- file picking ----------
        def pick_model(self):
            p = filedialog.askopenfilename(title="Your model", filetypes=[("EMD model", "*.emd *.EMD"), ("All files", "*.*")])
            if p:
                self.model_path.set(p)
                self.load_model()

        def pick_ref(self):
            p = filedialog.askopenfilename(title="Original game EMD", filetypes=[("EMD model", "*.emd *.EMD"), ("All files", "*.*")])
            if p:
                self.ref_path.set(p)
                self.load_model()

        def pick_src(self):
            p = filedialog.askopenfilename(title="New texture", filetypes=[
                ("Images / TIM", "*.png *.bmp *.jpg *.jpeg *.tga *.tim *.TIM"), ("All files", "*.*")])
            if p:
                self.src_path.set(p)
                self.load_src()

        def load_model(self):
            try:
                self.model_bytes = open(self.model_path.get(), 'rb').read()
                self.model_t = find_tim(self.model_bytes)
                self.pad = section_padding(self.model_bytes, self.model_t['end'])
                if self.ref_path.get():
                    self.ref_bytes = open(self.ref_path.get(), 'rb').read()
                    self.ref_t = find_tim(self.ref_bytes)
                else:
                    self.ref_bytes = self.ref_t = None
                self.ignore_holes = False
                if self.ref_t is not None and has_holes(self.ref_t):
                    key = self.ref_path.get()
                    if key not in self.hole_answers:
                        n = find_holes(self.ref_t)[1]
                        self.hole_answers[key] = messagebox.askyesno(
                            "RE1 Texture Tool",
                            "This original EMD has %d tiny see-through specks inside the artwork (for example the STARS "
                            "letters are invisible with it, as in the original Director's Cut).\n\n"
                            "Continue using it as it is?\n\n"
                            "YES: continue - the letters follow this original too.\n"
                            "NO: ignore those letter pixels - they are left exactly as they are in your model - and "
                            "continue with the rest of the original as usual." % n)
                    self.ignore_holes = not self.hole_answers[key]
                t = self.ref_t or self.model_t
                self.cx.set(t['cx']); self.cy.set(t['cy']); self.px.set(t['px']); self.py.set(t['py'])
                self.depth.set(t['bpp'])
                self.last_depth = t['bpp']
                self.status.set("Model loaded. Step 2: choose your new texture (PNG recommended).")
                h = self.health = texture_health(t)
                self.no_shadow = h['state'] == 'none'
                if h['state'] != 'ok':
                    # Keep-palette would copy the damage, so switch to the safe option and say why
                    self.mode.set('rebuild')
                    self.status.set("Transparency flags are missing/damaged - Rebuild palette selected. "
                                    "Add the original EMD to keep the shadow.")
                    messagebox.showwarning("RE1 Texture Tool", damaged_message(h))
                else:
                    self.mode.set('keep')
                    self.auto_mode()
                self.on_mode()                    # also refreshes the note under the file boxes
            except Exception as e:
                self.fail(e)

        def layout_ok(self):
            """False when the original EMD has another layout than the texture (e.g. another character's file)."""
            t = self.ref_t or self.model_t
            if t is None:
                return True
            src = self.src if self.replace.get() else (texture_to_rgba(self.model_t) if self.model_t is not None else None)
            if src is None or src.shape[:2] != (t['h'], t['w']):
                return True
            rm = row_map(t['clut'].shape[0], t['w'], t['h'])
            if not (t['clut'][rm, t['pix']] & 0x8000).any():
                return True
            return layout_similarity(t, src) >= LAYOUT_MATCH

        def auto_mode(self):
            """Pick Keep or Rebuild from how well the original palettes fit the texture. Silent: only the choice moves."""
            t = self.ref_t or self.model_t
            if t is None or self.health is None or self.health['state'] != 'ok' or self.depth.get() != t['bpp']:
                return
            src = self.src if self.replace.get() else texture_to_rgba(self.model_t)
            if src is None or src.shape[:2] != (t['h'], t['w']):
                return
            self.mode.set(suggest_mode(t, src))

        def update_pal_note(self):
            """Red only when the chosen option would really copy the damage; green when the choice is safe."""
            h = self.health
            if h is None:
                return
            have_ref = self.ref_t is not None
            who = "The original EMD you picked" if have_ref else "Your model"
            keep_clut = self.mode.get() == 'keep' and self.depth.get() in (4, 8)
            if h['state'] == 'ok' and self.ignore_holes and self.layout_ok():
                text = ("%s has see-through specks inside the artwork (like invisible STARS letters): those pixels are "
                        "ignored and left as they are in your texture; everything else follows the original." % who)
                colour = 'dark green'
            elif h['state'] == 'ok' and not self.layout_ok():
                text = ("%s has a different layout than this texture, so only its ground shadow is used; "
                        "the palettes are built from your image." % who)
                colour = 'dark green'
            elif h['state'] == 'ok':
                if keep_clut:
                    text = ("Palettes and shadow come from the original game EMD - good." if have_ref else
                            "Using your model's own palettes (they look healthy: %.1f%% semi-transparent). "
                            "Adding the original EMD is still the safest." % h['percent'])
                else:
                    text = "%s has a healthy shadow (%.1f%% semi-transparent); it is carried over into the result." % (who, h['percent'])
                colour = 'dark green'
            else:
                detail = ("has no shadow (semi-transparent) pixels" if h['state'] == 'none' else
                          "has damaged transparency flags (%.0f%% of pixels are semi-transparent)" % h['percent'])
                if keep_clut:
                    text = ("%s %s - the original game file keeps it to the shadow and dark colours. 'Keep game palette' would copy that damage, so "
                            "choose 'Rebuild palette' or put the untouched original EMD in 'Original game .EMD'." % (who, detail))
                    colour = 'red'
                else:
                    text = ("%s %s (the original game file keeps it to the shadow and dark colours), so there is no shadow to carry over. The new palettes "
                            "come from your image. To keep the shadow, put the untouched original EMD in 'Original game .EMD'."
                            % (who, detail))
                    colour = 'dark green'
            self.pal_label.configure(foreground=colour)
            self.pal_note.set(text)

        def load_src(self):
            try:
                self.src = load_source(self.src_path.get())
                t = self.ref_t or self.model_t
                if t and self.src.shape[:2] != (t['h'], t['w']):
                    if messagebox.askyesno("RE1 Texture Tool", "Your image is %dx%d but the texture is %dx%d.\n"
                                           "Resize it (nearest-neighbour)?" % (self.src.shape[1], self.src.shape[0], t['w'], t['h'])):
                        im = Image.fromarray(self.src).resize((t['w'], t['h']), Image.NEAREST)
                        self.src = np.array(im)
                self.auto_mode()
                self.on_mode()
            except Exception as e:
                self.fail(e)

        # ---------- actions ----------
        def on_replace(self):
            state = 'normal' if self.replace.get() else 'disabled'
            for w in self.src_widgets:
                w.configure(state=state)
            self.auto_mode()
            self.on_mode()

        def update_preview(self):
            self.result = None
            self.last_error = ''
            t = self.ref_t or self.model_t
            replacing = self.replace.get()
            # not replacing = the model's own current texture is the "new" image (only palettes / flags get fixed)
            src = self.src if replacing else (texture_to_rgba(self.model_t) if self.model_t is not None else None)
            try:
                if t is not None and src is not None:
                    self.result = convert(t, src, self.opts())
                    go = "  -> Save over my modded .EMD, or Save as new"
                    if replacing:
                        self.status.set("Ready. %d pixels differ from the original. %s%s" %
                                        (self.result['changed'], " ".join(self.result['notes']), go))
                    else:
                        mt = self.model_t
                        cur = mt['clut'][row_map(mt['clut'].shape[0], mt['w'], mt['h']), mt['pix']]
                        new = self.result['vals']
                        dist = np.sqrt(((vals_to_rgb(new) - vals_to_rgb(cur)) ** 2).sum(-1))
                        n_col, n_flag = int((dist > 0).sum()), int(((cur & 0x8000) != (new & 0x8000)).sum())
                        if n_col == 0 and n_flag == 0:
                            what = "Nothing would change - your model's texture already matches."
                        else:
                            what = ("Keeping your model's own texture: %d pixels change colour slightly (average %.1f out of 441), "
                                    "%d pixels get new transparency flags." % (n_col, dist[dist > 0].mean() if n_col else 0.0, n_flag))
                        tip = ""
                        if self.mode.get() == 'keep' and n_col > 100:
                            # Keep snaps colours that are not in the original palette; Rebuild may hold them exactly
                            alt = convert(t, src, dict(self.opts(), mode='rebuild'))
                            n_alt = int((vals_to_rgb(alt['vals']) != vals_to_rgb(cur)).any(axis=-1).sum())
                            if n_alt < n_col:
                                tip = ("  TIP: 'Rebuild palette' keeps your colours exactly (%d pixels change instead of %d); "
                                       "it is the less tested option in the game." % (n_alt, n_col))
                        self.status.set("Ready. %s %s%s%s" % (what, " ".join(self.result['notes']), tip, go))
            except Exception as e:
                self.last_error = str(e)
                self.status.set("Error: %s" % e)
            self.refresh()

        def tim_bytes(self):
            if self.result is None:
                raise ValueError(self.last_error or (
                    "Choose your model first." if self.model_t is None else
                    "Choose a new texture first (or untick 'Replace the texture with a new image')."))
            r = self.result
            if r['bpp'] == 24:
                return build_direct_tim(r['rgb24'], 24, self.px.get(), self.py.get())
            if r['bpp'] == 16:
                return build_direct_tim(r['vals'], 16, self.px.get(), self.py.get())
            return build_tim(r['clut'], r['pix'], self.cx.get(), self.cy.get(), self.px.get(), self.py.get(), r['bpp'])

        def build_emd(self):
            """The model with the new texture put in. Returns None if the user cancels at the shadow warning."""
            tim = self.tim_bytes()
            if self.result['bpp'] != self.model_t['bpp']:
                raise ValueError("Your model's texture is %d bit CLUT, but the result is %d bit. The game only reads the model's "
                                 "own depth inside an EMD - set 'Image data mode' back to %d bit CLUT (use the other modes with "
                                 "'Save texture as .TIM...')." % (self.model_t['bpp'], self.result['bpp'], self.model_t['bpp']))
            ph, pw = self.result['pix'].shape
            if texture_health(dict(clut=self.result['clut'], pix=self.result['pix'], w=pw, h=ph))['state'] == 'none' \
                    and not messagebox.askyesno("RE1 Texture Tool", SHADOW_WARNING + "\n\nSave anyway?"):
                return None
            b = self.model_bytes
            return b[:self.model_t['start']] + tim + b[self.model_t['end']:]

        def guard_original(self, path):
            """Never let an overwrite destroy the file that is serving as the 'original'."""
            if self.ref_path.get() and os.path.exists(path) and same_file(path, self.ref_path.get()):
                raise ValueError("Your model and the 'Original game .EMD' are the same file. Overwriting it would destroy "
                                 "your original - use 'Save as new .EMD...' instead.")

        def commit_emd(self, out, path):
            """Write the new EMD. Writing onto the loaded model is an overwrite (with a backup if ticked)."""
            mp = self.model_path.get()
            is_model = bool(mp) and os.path.exists(path) and same_file(path, mp)
            self.guard_original(path)
            bak = write_file_safely(path, out, self.backup.get() if is_model else False)
            if is_model:                      # keep the preview in step with the file on disk
                self.model_bytes = out
                self.model_t = find_tim(out)
                self.refresh()
            return is_model, bak

        def confirm_overwrite(self, path):
            return messagebox.askyesno("RE1 Texture Tool", "Overwrite your model?\n\n%s\n\n%s" % (
                path, "A backup copy (.bak) is made first." if self.backup.get() else "NO backup will be made."))

        def save_over(self):
            """Save over the loaded (modded) EMD."""
            try:
                mp = self.model_path.get()
                if not mp:
                    raise ValueError("Choose your model first.")
                self.guard_original(mp)          # refuse before asking anything
                out = self.build_emd()
                if out is None or not self.confirm_overwrite(mp):
                    return
                _, bak = self.commit_emd(out, mp)
                self.status.set("Your modded EMD was updated: %s%s" % (mp, ("   (backup: %s)" % bak) if bak else ""))
                messagebox.showinfo("RE1 Texture Tool", "Updated:\n%s%s" % (mp, ("\n\nBackup of the old file:\n%s" % bak) if bak else ""))
            except Exception as e:
                self.fail(e)

        def save_emd(self):
            """Save as a new EMD - your modded EMD is not touched."""
            try:
                out = self.build_emd()
                if out is None:
                    return
                stem = os.path.splitext(self.model_path.get())[0]
                p = filedialog.asksaveasfilename(title="Save new EMD", initialfile=os.path.basename(stem) + "_new.EMD",
                                                 defaultextension=".EMD")
                if not p:
                    return
                if os.path.exists(p) and same_file(p, self.model_path.get()):
                    self.guard_original(p)
                    if not self.confirm_overwrite(p):
                        return                # they picked their own model in the dialog: that is an overwrite
                is_model, bak = self.commit_emd(out, p)
                if is_model:
                    self.status.set("Your modded EMD was updated: %s%s" % (p, ("   (backup: %s)" % bak) if bak else ""))
                else:
                    self.status.set("Saved %s  (your modded EMD was not touched)" % p)
                messagebox.showinfo("RE1 Texture Tool", "Saved:\n%s" % p)
            except Exception as e:
                self.fail(e)

        def save_tim(self):
            try:
                tim = self.tim_bytes()
                p = filedialog.asksaveasfilename(title="Save TIM", defaultextension=".TIM", initialfile="texture.TIM")
                if p:
                    # same layout as RE123MV Editor's TIM: image data + the section's zero padding
                    if self.result['bpp'] in (4, 8):
                        open(p, 'wb').write(tim + b'\0' * self.pad)
                        self.status.set("Saved %s  (includes the %d padding bytes RE123MV Editor / the game expect)" % (p, self.pad))
                    else:
                        open(p, 'wb').write(tim)
                        self.status.set("Saved %s  (%d bit Direct TIM)" % (p, self.result['bpp']))
            except Exception as e:
                self.fail(e)

        def export_png(self):
            try:
                t = self.model_t
                if t is None:
                    raise ValueError("Choose your model first.")
                vals = t['clut'][row_map(t['clut'].shape[0], t['w'], t['h']), t['pix']]
                p = filedialog.asksaveasfilename(title="Save PNG", defaultextension=".png", initialfile="texture.png")
                if p:
                    rgb = vals_to_rgb(vals).astype(np.uint8)
                    Image.fromarray(np.dstack([rgb, np.where(vals == 0, 0, 255).astype(np.uint8)]), 'RGBA').save(p)
                    self.status.set("Saved %s - edit it, then load it as the new texture." % p)
            except Exception as e:
                self.fail(e)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    run_gui()
