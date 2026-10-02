#!/usr/bin/env python3
"""
RE1 Texture Tool - replace the character texture inside a Resident Evil 1 (PS1) .EMD model
with a PNG (recommended) or a .TIM, keeping the game's transparency (shadow) intact.

Needs: Python 3 + Pillow + numpy   (pip install pillow numpy)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

# =====================================================================================
#  CORE (no GUI) - TIM / EMD handling
# =====================================================================================


def parse_tim(d, o):
    """Parse an 8-bit-CLUT TIM located at offset o of bytes d."""
    if o + 20 > len(d) or struct.unpack_from('<I', d, o)[0] != 0x10:
        raise ValueError("Not a TIM")
    if struct.unpack_from('<I', d, o + 4)[0] != 9:
        raise ValueError("Only 8-bit CLUT TIMs are supported here")
    cb, cx, cy, cw, ch = struct.unpack_from('<IHHHH', d, o + 8)
    if cw != 256 or ch < 1 or cb != 12 + cw * ch * 2:
        raise ValueError("Unexpected CLUT size")
    clut = np.frombuffer(d, '<u2', cw * ch, o + 20).reshape(ch, cw).copy()
    p = o + 8 + cb
    pb, px, py, pw, ph = struct.unpack_from('<IHHHH', d, p)
    w = pw * 2
    if pb != 12 + w * ph or p + pb > len(d):
        raise ValueError("Unexpected pixel block size")
    pix = np.frombuffer(d, np.uint8, w * ph, p + 12).reshape(ph, w).copy()
    return dict(start=o, end=p + pb, clut=clut, cx=cx, cy=cy, pix=pix, px=px, py=py, w=w, h=ph)


def find_tim(d):
    """Find the (last) valid 8-bit TIM inside an EMD file."""
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
        raise ValueError("No 8-bit TIM texture found in this file.")
    return best


def build_tim(clut, pix, cx, cy, px, py):
    rows, cols = clut.shape
    h, w = pix.shape
    out = struct.pack('<II', 0x10, 9)
    out += struct.pack('<IHHHH', 12 + rows * cols * 2, cx, cy, cols, rows) + clut.astype('<u2').tobytes()
    out += struct.pack('<IHHHH', 12 + w * h, px, py, w // 2, h) + pix.astype(np.uint8).tobytes()
    return out


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
    return np.dstack([rgb, np.where(vals == 0, 0, 255).astype(np.uint8)])


def load_source(path):
    """Load the replacement image: PNG/JPG/BMP/... or .TIM -> RGBA uint8 array."""
    if path.lower().endswith('.tim'):
        return tim_to_rgba(open(path, 'rb').read())
    return np.array(Image.open(path).convert('RGBA'))


DEFAULT_OPTS = dict(mode='keep',                    # 'keep' = keep game palette | 'rebuild' = like the Photoshop plugin
                    except_black_translucent=False,  # rebuild mode only
                    black_transparent=True,          # rebuild mode only
                    keep_shadow=True,                # rebuild mode only (always on in 'keep' mode)
                    honor_alpha=False)               # treat transparent pixels of the new image as transparent


def convert(ref, src, opts):
    """
    ref  : parsed TIM whose palettes / transparency bits are the reference (the original game texture)
    src  : RGBA uint8 array (h, w, 4) with the new artwork
    Returns dict(pix, clut, vals, notes, changed)
    """
    clut_ref, ref_pix = ref['clut'], ref['pix']
    rows, h, w = clut_ref.shape[0], ref['h'], ref['w']
    if src.shape[:2] != (h, w):
        raise ValueError("The new image must be %dx%d pixels (it is %dx%d)." % (w, h, src.shape[1], src.shape[0]))
    rm = row_map(rows, w, h)
    ref_vals = clut_ref[rm, ref_pix]
    ref_rgb = vals_to_rgb(ref_vals)
    ref_stp = (ref_vals & 0x8000) != 0
    notes = []

    rgb = (src[..., :3].astype(np.int32) >> 3) << 3          # PS1 keeps 5 bits per channel
    transp = src[..., 3] < 128
    if opts['honor_alpha']:
        rgb = np.where(transp[..., None], 0, rgb)
    else:
        kept = transp & (ref_vals != 0)
        rgb = np.where(transp[..., None], ref_rgb, rgb)
        if kept.any():
            notes.append("%d transparent pixels in your image were opaque in the original game texture; "
                         "the original pixel was kept." % int(kept.sum()))

    new_pix, new_clut = ref_pix.copy(), clut_ref.copy()

    if opts['mode'] == 'keep':
        todo = ~np.all(rgb == ref_rgb, axis=-1) & ~ref_stp
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
        prot = ref_stp if opts['keep_shadow'] else np.zeros_like(ref_stp)
        stp_bit = 1 if opts['except_black_translucent'] else 0
        for r in range(rows):
            cm = rm == r
            pm = prot & cm
            reserved = sorted(set(ref_pix[pm].tolist()))
            avail = [i for i in range(256) if i not in set(reserved)]
            sel = cm & ~pm
            row = np.zeros(256, np.uint16)
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
        notes.append("Palettes rebuilt from your image (%s)." %
                     ("shadow pixels from the original kept" if opts['keep_shadow'] else "shadow NOT preserved"))

    vals = new_clut[rm, new_pix]
    return dict(pix=new_pix, clut=new_clut, vals=vals, notes=notes, changed=int((vals != ref_vals).sum()))


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
            self.zoom = tk.IntVar(value=2)
            self.overlay = tk.BooleanVar(value=False)
            self.status = tk.StringVar(value="Step 1: choose your model (.EMD).")
            self.model_bytes = self.ref_bytes = None
            self.model_t = self.ref_t = self.src = self.result = None
            self.photos = {}
            self.build()

        # ---------- layout ----------
        def build(self):
            r = self.root
            top = ttk.LabelFrame(r, text=" Files ")
            top.pack(fill='x', padx=8, pady=6)
            rows = [("Your model (.EMD):", self.model_path, self.pick_model),
                    ("Original game .EMD (optional):", self.ref_path, self.pick_ref),
                    ("New texture (PNG recommended, or .TIM):", self.src_path, self.pick_src)]
            for i, (label, var, cmd) in enumerate(rows):
                ttk.Label(top, text=label).grid(row=i, column=0, sticky='w', padx=6, pady=2)
                ttk.Entry(top, textvariable=var, width=70).grid(row=i, column=1, padx=4)
                ttk.Button(top, text="Browse...", command=cmd).grid(row=i, column=2, padx=4)
            ttk.Label(top, foreground='gray', text="Optional file = where palettes and the shadow come from. "
                      "Leave empty to use your model's own.").grid(row=3, column=0, columnspan=3, sticky='w', padx=6)

            mid = ttk.Frame(r)
            mid.pack(fill='both', expand=True, padx=8)
            left = ttk.Frame(mid)
            left.pack(side='left', fill='both', expand=True)
            bar = ttk.Frame(left)
            bar.pack(fill='x')
            ttk.Label(bar, text="Zoom:").pack(side='left')
            for z in (1, 2, 3):
                ttk.Radiobutton(bar, text="%dx" % z, value=z, variable=self.zoom, command=self.refresh).pack(side='left')
            ttk.Checkbutton(bar, text="Highlight semi-transparent (shadow) areas", variable=self.overlay,
                            command=self.refresh).pack(side='left', padx=10)
            self.nb = ttk.Notebook(left)
            self.nb.pack(fill='both', expand=True)
            self.labels = {}
            for key, title in (('now', "In your model now"), ('new', "New result"), ('ref', "Original (reference)")):
                f = ttk.Frame(self.nb)
                self.nb.add(f, text=title)
                self.labels[key] = ttk.Label(f, anchor='center')
                self.labels[key].pack(fill='both', expand=True)

            right = ttk.LabelFrame(mid, text=" Settings ")
            right.pack(side='right', fill='y', padx=(8, 0))
            ttk.Label(right, text="Palette handling").pack(anchor='w', padx=6, pady=(4, 0))
            ttk.Radiobutton(right, text="Keep game palette (recommended)", value='keep', variable=self.mode,
                            command=self.on_mode).pack(anchor='w', padx=14)
            ttk.Radiobutton(right, text="Rebuild palette (like the Photoshop plugin)", value='rebuild',
                            variable=self.mode, command=self.on_mode).pack(anchor='w', padx=14)
            ttk.Separator(right).pack(fill='x', pady=6)
            ttk.Label(right, text="Image data mode").pack(anchor='w', padx=6)
            ttk.Radiobutton(right, text="8 bit CLUT (the mode RE1 uses)", value=1, variable=tk.IntVar(value=1)).pack(
                anchor='w', padx=14)
            ttk.Separator(right).pack(fill='x', pady=6)
            g = ttk.Frame(right)
            g.pack(padx=6)
            ttk.Label(g, text="CLUT section (VRAM)").grid(row=0, column=0, columnspan=4, sticky='w')
            ttk.Label(g, text="X").grid(row=1, column=0)
            ttk.Entry(g, textvariable=self.cx, width=5).grid(row=1, column=1)
            ttk.Label(g, text="Y").grid(row=1, column=2)
            ttk.Entry(g, textvariable=self.cy, width=5).grid(row=1, column=3)
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
            ttk.Checkbutton(right, text="Treat transparent pixels of my\nimage as transparent", variable=self.honor_alpha,
                            command=self.update_preview).pack(anchor='w', padx=14, pady=(6, 0))
            ttk.Separator(right).pack(fill='x', pady=6)
            ttk.Button(right, text="Default", command=self.defaults).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save current texture as PNG...", command=self.export_png).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save new .EMD...", command=self.save_emd).pack(fill='x', padx=10, pady=2)
            ttk.Button(right, text="Save texture as .TIM...", command=self.save_tim).pack(fill='x', padx=10, pady=2)

            ttk.Label(r, textvariable=self.status, relief='sunken', anchor='w', wraplength=1000).pack(
                fill='x', side='bottom', padx=8, pady=6)
            self.on_mode()

        # ---------- helpers ----------
        def on_mode(self):
            state = 'normal' if self.mode.get() == 'rebuild' else 'disabled'
            for cb in (self.cb_ebt, self.cb_bt, self.cb_ks):
                cb.configure(state=state)
            self.update_preview()

        def defaults(self):
            self.mode.set('keep')
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
                        keep_shadow=self.keep_shadow.get(), honor_alpha=self.honor_alpha.get())

        def show(self, key, vals):
            if vals is None:
                self.labels[key].configure(image='', text="(nothing to show yet)")
                return
            img = render_rgba(vals, self.overlay.get())
            z = self.zoom.get()
            img = img.resize((img.width * z, img.height * z), Image.NEAREST)
            self.photos[key] = ImageTk.PhotoImage(img)
            self.labels[key].configure(image=self.photos[key], text='')

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
                if self.ref_path.get():
                    self.ref_bytes = open(self.ref_path.get(), 'rb').read()
                    self.ref_t = find_tim(self.ref_bytes)
                else:
                    self.ref_bytes = self.ref_t = None
                t = self.ref_t or self.model_t
                self.cx.set(t['cx']); self.cy.set(t['cy']); self.px.set(t['px']); self.py.set(t['py'])
                self.clut_size.configure(text="Size: %d x %d" % (t['clut'].shape[1], t['clut'].shape[0]))
                self.pix_size.configure(text="Size: %d x %d (words: %d)" % (t['w'], t['h'], t['w'] // 2))
                self.status.set("Model loaded. Step 2: choose your new texture (PNG recommended).")
                self.update_preview()
            except Exception as e:
                self.fail(e)

        def load_src(self):
            try:
                self.src = load_source(self.src_path.get())
                t = self.ref_t or self.model_t
                if t and self.src.shape[:2] != (t['h'], t['w']):
                    if messagebox.askyesno("RE1 Texture Tool", "Your image is %dx%d but the texture is %dx%d.\n"
                                           "Resize it (nearest-neighbour)?" % (self.src.shape[1], self.src.shape[0], t['w'], t['h'])):
                        im = Image.fromarray(self.src).resize((t['w'], t['h']), Image.NEAREST)
                        self.src = np.array(im)
                self.update_preview()
            except Exception as e:
                self.fail(e)

        # ---------- actions ----------
        def update_preview(self):
            self.result = None
            t = self.ref_t or self.model_t
            try:
                if t is not None and self.src is not None:
                    self.result = convert(t, self.src, self.opts())
                    self.status.set("Ready. %d pixels differ from the original. %s  -> Save new .EMD" %
                                    (self.result['changed'], " ".join(self.result['notes'])))
            except Exception as e:
                self.status.set("Error: %s" % e)
            self.refresh()

        def tim_bytes(self):
            if self.result is None:
                raise ValueError("Choose a model and a new texture first.")
            return build_tim(self.result['clut'], self.result['pix'], self.cx.get(), self.cy.get(), self.px.get(), self.py.get())

        def save_emd(self):
            try:
                tim = self.tim_bytes()
                b = self.model_bytes
                out = b[:self.model_t['start']] + tim + b[self.model_t['end']:]
                stem = os.path.splitext(self.model_path.get())[0]
                p = filedialog.asksaveasfilename(title="Save new EMD", initialfile=os.path.basename(stem) + "_new.EMD",
                                                 defaultextension=".EMD")
                if p:
                    open(p, 'wb').write(out)
                    self.status.set("Saved %s  (your original was not touched)" % p)
                    messagebox.showinfo("RE1 Texture Tool", "Saved:\n%s" % p)
            except Exception as e:
                self.fail(e)

        def save_tim(self):
            try:
                tim = self.tim_bytes()
                p = filedialog.asksaveasfilename(title="Save TIM", defaultextension=".TIM", initialfile="texture.TIM")
                if p:
                    open(p, 'wb').write(tim)
                    self.status.set("Saved %s" % p)
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
