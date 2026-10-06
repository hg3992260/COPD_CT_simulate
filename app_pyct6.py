#!/usr/bin/env python3
"""
Lung Volume Change Heatmap Viewer — Professional Edition
Aligned with: Shima et al. (2023) Thorax 78(4):344-353
developed by christ.paul90@gmail.com
"""

import os, sys, glob
import numpy as np
import SimpleITK as sitk
import pydicom

from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QFileDialog,
    QLabel, QRadioButton, QButtonGroup, QProgressBar, QMessageBox,
    QWidget, QSlider, QSizePolicy, QSpacerItem, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QFont, QColor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyCt6 import (
    CMainWindow, CButton, CLineEdit, CLabel, CFrame,
    set_color_theme, set_appearance_mode
)

# ── Design Tokens ──
ACCENT      = "#0078d7"
ACCENT_DIM  = "#005a9e"
TEXT_PRIMARY = ("#1a1a1a", "#e0e0e0")
TEXT_SECOND  = ("#555555", "#999999")
TEXT_MUTED   = ("#888888", "#666666")
BG_CARD     = ("#f8f8f8", "#2a2a2a")
BG_DARK     = ("#1a1a2e", "#1a1a2e")
BORDER      = ("#d0d0d0", "#404040")
STAT_BLUE   = ("#0078d7", "#60cdff")
STAT_GREEN  = ("#0f7b0f", "#6ccb5f")
STAT_RED    = ("#c42b1c", "#ff99a4")
STAT_AMBER  = ("#9d5d00", "#fce100")
STAT_PURPLE = ("#6b3fa0", "#b48adf")

# ── Typography ──
FONT_FAMILY = "Segoe UI"

def FONT(sz=11, bold=False):
    f = QFont(FONT_FAMILY, sz)
    if bold: f.setBold(True)
    return f

# ── Mesh presets ──
TARGET_SPACINGS = [
    (80.0, 120.0), (50.0, 75.0), (35.0, 55.0), (24.0, 36.0), (18.0, 27.0),
]

def build_mesh_presets(size_x, size_y, size_z, sp_x, sp_y, sp_z):
    presets = {}
    for level, (tgt_xy, tgt_z) in enumerate(TARGET_SPACINGS, 1):
        nx = max(2, int(np.ceil(size_x * sp_x / tgt_xy)))
        ny = max(2, int(np.ceil(size_y * sp_y / tgt_xy)))
        nz = max(2, int(np.ceil(size_z * sp_z / tgt_z)))
        presets[level] = [nx, ny, nz]
    return presets

# ── Core computation functions (unchanged) ──

def load_ct_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")
    series_id = series_ids[0]
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    reader.SetFileNames(dicom_files)
    return reader.Execute()

def segment_lungs(ct_image, lower_thresh=-1000, upper_thresh=-400):
    binary = sitk.BinaryThreshold(ct_image,
        lowerThreshold=lower_thresh, upperThreshold=upper_thresh,
        insideValue=1, outsideValue=0)
    binary = sitk.BinaryMorphologicalOpening(binary, kernelRadius=[2, 2, 2])
    binary = sitk.BinaryFillhole(binary)
    connected = sitk.ConnectedComponent(binary)
    relabel = sitk.RelabelComponent(connected, minimumObjectSize=1000)
    return sitk.BinaryThreshold(relabel, lowerThreshold=1, upperThreshold=2,
                                insideValue=1, outsideValue=0)

def register_images(fixed_image, moving_image, mesh_size=(8, 8, 4), fixed_mask=None):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    if fixed_mask is not None:
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                              maximumNumberOfCorrections=5,
                              maximumNumberOfFunctionEvaluations=1000,
                              costFunctionConvergenceFactor=1e+7)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0])
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    tx = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
    reg.SetInitialTransform(tx, inPlace=False)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    return reg.Execute(fixed_image, moving_image)

def compute_volume_change_map(fixed_image, transform):
    disp_field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64,
        fixed_image.GetSize(), fixed_image.GetOrigin(),
        fixed_image.GetSpacing(), fixed_image.GetDirection())
    disp_field = sitk.Cast(disp_field, sitk.sitkVectorFloat32)
    jacobian = sitk.DisplacementFieldJacobianDeterminant(disp_field, useImageSpacing=True)
    return jacobian * 100.0

def slice_image_to_rgba(slice_2d, mask_2d, vmin, vmax, cmap_name='RdBu_r'):
    cmap = plt.get_cmap(cmap_name)
    valid = mask_2d > 0
    normalized = np.clip((slice_2d - vmin) / (vmax - vmin + 1e-10), 0, 1)
    rgba = cmap(normalized)
    rgba[~valid] = [0, 0, 0, 0]
    return (rgba * 255).astype(np.uint8)

def rgba_to_qimage(rgba):
    h, w, _ = rgba.shape
    return QImage(rgba.data.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)

def render_overlay(ct_slice, heat_slice, mask_slice, vmin, vmax, wl=-600, ww=1500):
    cmap = plt.get_cmap('RdBu_r')
    ct_min = wl - ww / 2
    ct_max = wl + ww / 2
    ct_clipped = np.clip((ct_slice.astype(np.float32) - ct_min) / (ct_max - ct_min + 1e-10), 0, 1)
    ct_gray = (ct_clipped * 255).astype(np.uint8)
    valid = mask_slice > 0
    h2d_norm = np.clip((heat_slice - vmin) / (vmax - vmin + 1e-10), 0, 1)
    color_overlay = (cmap(h2d_norm) * 255).astype(np.uint8)
    alpha = (0.65 * valid.astype(np.float32))[:, :, None]
    ct_rgba = np.dstack([ct_gray, ct_gray, ct_gray, np.full_like(ct_gray, 255, dtype=np.uint8)])
    result = ((1 - alpha) * ct_rgba + alpha * color_overlay).astype(np.uint8)
    result[:, :, 3] = 255
    return result

def get_dcm_meta(dicom_dir):
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, '*.dcm')))
    if not dcm_files:
        dcm_files = sorted([os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                           if os.path.isfile(os.path.join(dicom_dir, f))])
    ds = pydicom.dcmread(dcm_files[0], force=True)
    pid = str(getattr(ds, 'PatientID', 'UNKNOWN')).strip()
    thickness = str(getattr(ds, 'SliceThickness', '0'))
    instance_numbers = set()
    for f in dcm_files[:2000]:
        try:
            d = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            ino = getattr(d, 'InstanceNumber', None)
            if ino is not None:
                instance_numbers.add(int(ino))
        except Exception:
            pass
    return {'patient_id': pid, 'slice_thickness': thickness,
            'num_slices': len(instance_numbers) if instance_numbers else len(dcm_files),
            'file_count': len(dcm_files)}

# ── Computation Worker ──

class ComputeWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, insp_dir, exp_dir, mesh=(8, 8, 4)):
        super().__init__()
        self.insp_dir = insp_dir
        self.exp_dir = exp_dir
        self.mesh = mesh

    def run(self):
        try:
            self.progress.emit("Loading inspiratory CT ...")
            insp_img = load_ct_series(self.insp_dir)
            self.progress.emit("Loading expiratory CT ...")
            exp_img = load_ct_series(self.exp_dir)
            self.progress.emit("Segmenting lung parenchyma ...")
            lung_mask = segment_lungs(insp_img, -1000, -400)
            self.progress.emit("Registering dual-phase lung tissue (please wait) ...")
            insp_lung   = sitk.Cast(insp_img, sitk.sitkFloat32)
            exp_lung    = sitk.Cast(exp_img, sitk.sitkFloat32)
            lung_mask_f = sitk.Cast(lung_mask, sitk.sitkFloat32)
            insp_lung   = sitk.Multiply(insp_lung, lung_mask_f)
            exp_lung    = sitk.Multiply(exp_lung, lung_mask_f)
            transform = register_images(insp_lung, exp_lung,
                                        mesh_size=self.mesh, fixed_mask=lung_mask)
            self.progress.emit("Computing volume change map (J x 100) ...")
            vol_change = compute_volume_change_map(insp_lung, transform)
            self.progress.emit("Masking non-lung voxels ...")
            vol_change = sitk.Mask(vol_change, lung_mask, outsideValue=0.0, maskingValue=0.0)
            vol_arr = sitk.GetArrayFromImage(vol_change)
            mask_arr = sitk.GetArrayFromImage(lung_mask)
            valid = mask_arr > 0
            vmin_val = float(np.percentile(vol_arr[valid], 1))
            vmax_val = float(np.percentile(vol_arr[valid], 99))
            voxel_vol = float(np.prod(insp_img.GetSpacing()))
            lung_voxels = int(mask_arr.sum())
            lung_vol_ml = lung_voxels * voxel_vol / 1000.0

            self.progress.emit("Computing mismatch / air trapping ...")
            exp_warped = sitk.Resample(sitk.Cast(exp_img, sitk.sitkFloat32),
                                        insp_img, transform, sitk.sitkLinear, -1024.0)
            insp_full = sitk.GetArrayFromImage(insp_img)
            exp_w = sitk.GetArrayFromImage(exp_warped)
            mismatch_mask = (insp_full > -950) & (exp_w < -856) & (mask_arr.astype(bool))
            mismatch_ml = int(mismatch_mask.sum()) * voxel_vol / 1000.0
            trap_mask = (exp_w >= -950) & (exp_w <= -856) & (mask_arr.astype(bool))
            trap_ml = int(trap_mask.sum()) * voxel_vol / 1000.0

            self.progress.emit("Computing statistics ...")
            vc_valid = vol_arr[valid]
            stats = {
                'mean_vc': float(np.mean(vc_valid)),
                'std_vc':  float(np.std(vc_valid)),
                'p5_vc':   float(np.percentile(vc_valid, 5)),
                'p95_vc':  float(np.percentile(vc_valid, 95)),
                'low_vent_pct':  float(np.sum(vc_valid < 95) / len(vc_valid) * 100),
                'high_vent_pct': float(np.sum(vc_valid > 105) / len(vc_valid) * 100),
            }

            self.progress.emit("Preparing CT overlay data ...")
            ct_arr = sitk.GetArrayFromImage(insp_img)
            spacing = tuple(float(s) for s in insp_img.GetSpacing())
            self.progress.emit(f"Done — Lung volume: {lung_vol_ml:.1f} mL")
            self.finished.emit({
                'vol_arr': vol_arr, 'mask_arr': mask_arr, 'ct_arr': ct_arr,
                'spacing': spacing, 'vmin': vmin_val, 'vmax': vmax_val,
                'shape': vol_arr.shape, 'lung_vol_ml': lung_vol_ml,
                'lung_voxels': lung_voxels, 'mesh': self.mesh,
                'mismatch_ml': mismatch_ml, 'trap_ml': trap_ml, **stats,
            })
        except Exception as e:
            self.error.emit(str(e))


# ── Helper: stat card widget ──

def make_stat_card(label_text, value_text, color_token):
    """Create a compact stat card: [label] [value] with colored value."""
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background-color: transparent;
            border: 1px solid {BORDER[0]};
            border-radius: 6px;
            padding: 4px 8px;
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(1)

    lbl = QLabel(label_text)
    lbl.setFont(FONT(8))
    lbl.setStyleSheet(f"color: {TEXT_SECOND[1]}; border: none; background: transparent;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(lbl)

    val = QLabel(value_text)
    val.setFont(FONT(13, bold=True))
    val.setStyleSheet(f"color: {color_token[0]}; border: none; background: transparent;")
    val.setAlignment(Qt.AlignmentFlag.AlignCenter)
    val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(val)

    return frame, val


# ── Helper: attach an inner layout to a CFrame ──

def frame_body(frame):
    """Return a child QWidget managed by the CFrame's built-in layout.

    PyCt6's CFrame already installs its own layout, so creating a second
    ``QHBoxLayout(frame)`` + ``frame.setLayout(...)`` is silently ignored and
    the inner widgets collapse to (0, 0).  This host widget gives the frame's
    layout exactly one expanding child whose own layout is free to be set.
    """
    body = QWidget(frame)
    body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    lay = frame.layout()
    if lay is not None:
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.setAlignment(Qt.AlignmentFlag(0))
        lay.addWidget(body)
    return body


# ── Slice Viewer (enhanced) ──

class SliceViewer(QWidget):
    def __init__(self, master):
        super().__init__(master)
        self.data = None
        self.current_axis = 'coronal'
        self.overlay_enabled = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setStyleSheet(
            "background-color: #0a0a14; border: 1px solid #333; border-radius: 6px;")

        self.info_label = QLabel("Ready")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(FONT(10))
        self.info_label.setStyleSheet(f"color: {TEXT_SECOND[0]};")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #444; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078d7; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d7; border-radius: 2px;
            }
        """)
        self.slider.valueChanged.connect(self._on_slide)

        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.info_label)
        layout.addWidget(self.slider)

    def set_data(self, data):
        self.data = data
        self.slider.setEnabled(True)
        self._update_slider_range()

    def toggle_overlay(self, enabled):
        self.overlay_enabled = enabled
        self._render_current()

    def _update_slider_range(self):
        if self.data is None: return
        shape = {'axial': 0, 'coronal': 1, 'sagittal': 2}[self.current_axis]
        n = self.data['vol_arr'].shape[shape]
        self.slider.setRange(0, n - 1)
        self.slider.setValue(n // 2)

    def switch_axis(self, axis):
        self.current_axis = axis
        self._update_slider_range()
        self._render_current()

    def _on_slide(self):
        self._render_current()

    def _render_current(self):
        if self.data is None: return
        idx = self.slider.value()
        vol = self.data['vol_arr']
        mask = self.data['mask_arr']
        ct   = self.data['ct_arr']
        sx, sy, sz = self.data['spacing']
        vmin, vmax = self.data['vmin'], self.data['vmax']
        names = {'axial': 'Axial', 'coronal': 'Coronal', 'sagittal': 'Sagittal'}

        if self.current_axis == 'axial':
            s2d = vol[idx, :, :]; m2d = mask[idx, :, :]; ct2d = ct[idx, :, :]
            h_sp, w_sp = sy, sx; n = vol.shape[0]
        elif self.current_axis == 'coronal':
            s2d = np.flipud(vol[:, idx, :]); m2d = np.flipud(mask[:, idx, :]); ct2d = np.flipud(ct[:, idx, :])
            h_sp, w_sp = sz, sx; n = vol.shape[1]
        else:
            s2d = np.flipud(vol[:, :, idx]); m2d = np.flipud(mask[:, :, idx]); ct2d = np.flipud(ct[:, :, idx])
            h_sp, w_sp = sz, sy; n = vol.shape[2]

        self.info_label.setText(
            f"{names[self.current_axis]}  Slice {idx+1}/{n}  |  "
            f"{vmin:.1f}%  < 100 = no change <  {vmax:.1f}%")

        rgba = render_overlay(ct2d, s2d, m2d, vmin, vmax) if self.overlay_enabled \
               else slice_image_to_rgba(s2d, m2d, vmin, vmax)

        pix_h, pix_w = rgba.shape[:2]
        aspect = (pix_h * h_sp) / (pix_w * w_sp + 1e-10)
        lw = self.image_label.width() or 600
        lh = self.image_label.height() or 450
        dw, dh = (int(lh * aspect), lh) if lw / lh > aspect else (lw, int(lw / aspect))
        qimg = rgba_to_qimage(rgba)
        pm = QPixmap.fromImage(qimg).scaled(max(dw,1), max(dh,1),
                                            Qt.AspectRatioMode.IgnoreAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pm)


# ── Main Window (Professional Layout) ──

LOGO_PATH = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(__file__)), 'logo.png')
DEFAULT_ICON = LOGO_PATH if os.path.exists(LOGO_PATH) else None

class MainWindow(CMainWindow):
    AXIS_MAP = {'Coronal': 'coronal', 'Sagittal': 'sagittal', 'Axial': 'axial'}

    def __init__(self):
        super().__init__(width=1200, height=820,
                         title="Lung Volume Change Heatmap — Shima et al. (2023)",
                         icon=DEFAULT_ICON)
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(16, 12, 16, 12)
        self.mesh_presets = build_mesh_presets(512, 512, 46, 0.574, 0.574, 5.0)

        # ═══════ Title Bar ═══════
        title_frame = CFrame(self, width=1160, height=48, corner_radius=8,
                             background_color=BG_CARD)
        title_inner = QHBoxLayout(frame_body(title_frame))
        title_inner.setContentsMargins(16, 6, 16, 6)

        app_title = CLabel(title_frame, width=500, height=30,
                           text="Lung Volume Change Analysis",
                           font_size=18, font_style="bold")
        app_title.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_inner.addWidget(app_title)

        subtitle = CLabel(title_frame, width=500, height=30,
                          text="V(%) = J x 100  |  100 = no change, <100 = shrink, >100 = expand",
                          font_size=10)
        subtitle.label().setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        subtitle.label().setStyleSheet(f"color: {TEXT_SECOND[1]};")
        title_inner.addWidget(subtitle)

        self.main_layout.addWidget(title_frame)

        # ═══════ Input Section ═══════
        input_frame = CFrame(self, width=1160, height=80, corner_radius=8,
                             background_color=BG_CARD)
        input_inner = QHBoxLayout(frame_body(input_frame))
        input_inner.setContentsMargins(12, 8, 12, 8)
        input_inner.setSpacing(12)

        # Inspiration path
        insp_col = QVBoxLayout()
        insp_col.setSpacing(4)
        insp_hdr = CLabel(input_frame, text="Inspiratory Phase (CT)", width=380, height=18,
                          font_size=11, font_style="bold")
        insp_hdr.label().setAlignment(Qt.AlignmentFlag.AlignLeft)
        insp_col.addWidget(insp_hdr)

        insp_row = QHBoxLayout()
        insp_row.setSpacing(6)
        self.insp_edit = CLineEdit(input_frame, width=340, height=28,
                                   placeholder_text="Select inspiratory DICOM directory",
                                   font_size=10,
                                   text_color=TEXT_PRIMARY,
                                   placeholder_text_color=TEXT_MUTED)
        insp_btn = CButton(input_frame, width=80, height=28, text="Browse", font_size=10,
                           command=lambda: self._browse(self.insp_edit))
        insp_row.addWidget(self.insp_edit)
        insp_row.addWidget(insp_btn)
        insp_col.addLayout(insp_row)
        input_inner.addLayout(insp_col)

        # Expiration path
        exp_col = QVBoxLayout()
        exp_col.setSpacing(4)
        exp_hdr = CLabel(input_frame, text="Expiratory Phase (CT)", width=380, height=18,
                         font_size=11, font_style="bold")
        exp_hdr.label().setAlignment(Qt.AlignmentFlag.AlignLeft)
        exp_col.addWidget(exp_hdr)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(6)
        self.exp_edit = CLineEdit(input_frame, width=340, height=28,
                                  placeholder_text="Select expiratory DICOM directory",
                                  font_size=10,
                                  text_color=TEXT_PRIMARY,
                                  placeholder_text_color=TEXT_MUTED)
        exp_btn = CButton(input_frame, width=80, height=28, text="Browse", font_size=10,
                          command=lambda: self._browse(self.exp_edit))
        exp_row.addWidget(self.exp_edit)
        exp_row.addWidget(exp_btn)
        exp_col.addLayout(exp_row)
        input_inner.addLayout(exp_col)

        # Compute button
        btn_col = QVBoxLayout()
        btn_col.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.calc_btn = CButton(input_frame, width=140, height=40, text="  Compute  ",
                                font_size=13, font_style="bold",
                                background_color=ACCENT,
                                hover_color=ACCENT_DIM,
                                command=self._start_compute)
        btn_col.addWidget(self.calc_btn)
        input_inner.addLayout(btn_col)

        self.main_layout.addWidget(input_frame)

        # ═══════ Progress ═══════
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: none; background: #333; border-radius: 1px; }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 1px; }}
        """)
        self.progress_label = CLabel(self, width=1160, height=16, text="", font_size=10)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.progress_label)

        # ═══════ Mesh / Settings Row ═══════
        settings_frame = CFrame(self, width=1160, height=50, corner_radius=8,
                                background_color=BG_CARD)
        settings_inner = QHBoxLayout(frame_body(settings_frame))
        settings_inner.setContentsMargins(12, 6, 12, 6)
        settings_inner.setSpacing(12)

        mesh_lbl = CLabel(settings_frame, text="Registration Precision:", width=140, height=22,
                          font_size=10, font_style="bold")
        settings_inner.addWidget(mesh_lbl)

        self.mesh_slider = QSlider(Qt.Orientation.Horizontal)
        self.mesh_slider.setRange(1, 5)
        self.mesh_slider.setValue(3)
        self.mesh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.mesh_slider.setTickInterval(1)
        self.mesh_slider.setFixedWidth(160)
        settings_inner.addWidget(self.mesh_slider)

        self.mesh_value_label = CLabel(settings_frame, width=120, height=22,
                                       text="3  [8x8x4]", font_size=10, font_style="bold")
        settings_inner.addWidget(self.mesh_value_label)

        self.mesh_spacing_label = CLabel(settings_frame, width=280, height=22,
                                         text="XY ~36.7mm  Z ~57.5mm", font_size=9)
        self.mesh_spacing_label.label().setStyleSheet(f"color: {TEXT_MUTED[1]};")
        settings_inner.addWidget(self.mesh_spacing_label)

        settings_inner.addSpacing(20)

        mesh_hint = CLabel(settings_frame, width=300, height=22,
                           text="Higher = finer grid, more accurate, slower",
                           font_size=9)
        mesh_hint.label().setStyleSheet(f"color: {TEXT_MUTED[1]};")
        settings_inner.addWidget(mesh_hint)
        settings_inner.addStretch()

        self.main_layout.addWidget(settings_frame)
        self.mesh_slider.valueChanged.connect(self._on_mesh_changed)

        # ═══════ View Controls ═══════
        controls_frame = CFrame(self, width=1160, height=36, corner_radius=6,
                                background_color=BG_CARD)
        controls_inner = QHBoxLayout(frame_body(controls_frame))
        controls_inner.setContentsMargins(12, 4, 12, 4)
        controls_inner.setSpacing(10)

        view_lbl = CLabel(controls_frame, text="View:", width=40, height=24,
                          font_size=11, font_style="bold")
        controls_inner.addWidget(view_lbl)

        self.view_btns = QButtonGroup(self)
        for name in ['Coronal', 'Sagittal', 'Axial']:
            rb = QRadioButton(name)
            rb.setFont(FONT(10))
            rb.setStyleSheet(f"color: {TEXT_PRIMARY[1]}; padding: 2px 6px;")
            self.view_btns.addButton(rb)
            controls_inner.addWidget(rb)
            rb.toggled.connect(self._on_axis_changed)
        self.view_btns.buttons()[0].setChecked(True)

        controls_inner.addSpacing(16)

        self.overlay_btn = CButton(controls_frame, width=120, height=24,
                                   text="CT Overlay", font_size=10, font_style="bold")
        self.overlay_btn.button().setCheckable(True)
        self.overlay_btn.button().toggled.connect(lambda c: self._on_overlay_toggled(c))
        controls_inner.addWidget(self.overlay_btn)

        export_btn = CButton(controls_frame, width=100, height=24,
                             text="Export PNG", font_size=10)
        export_btn.button().clicked.connect(self._export_image)
        controls_inner.addWidget(export_btn)

        self.colorbar_info = CLabel(controls_frame, width=440, height=24, text="", font_size=10)
        controls_inner.addWidget(self.colorbar_info)
        controls_inner.addStretch()

        credit = CLabel(controls_frame, width=200, height=24,
                        text="christ.paul90@gmail.com", font_size=9)
        credit.label().setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        credit.label().setStyleSheet(f"color: {TEXT_MUTED[1]};")
        controls_inner.addWidget(credit)

        self.main_layout.addWidget(controls_frame)

        # ═══════ Statistics Panel (Card Grid) ═══════
        stats_outer = CFrame(self, width=1160, height=80, corner_radius=8,
                             background_color=BG_CARD)
        stats_grid = QHBoxLayout(frame_body(stats_outer))
        stats_grid.setContentsMargins(12, 6, 12, 6)
        stats_grid.setSpacing(8)

        stat_defs = [
            ("Lung Volume",   "--",  STAT_BLUE,   "stat_lung"),
            ("Mean V(%)",     "--",  STAT_GREEN,  "stat_mean"),
            ("SD",            "--",  STAT_PURPLE, "stat_sd"),
            ("P5",            "--",  STAT_RED,    "stat_p5"),
            ("P95",           "--",  STAT_AMBER,  "stat_p95"),
            ("Low Vent <95%", "--",  STAT_RED,    "stat_low"),
            ("High Vent >105%","--", STAT_AMBER,  "stat_high"),
            ("Mismatch",      "--",  STAT_RED,    "stat_mismatch"),
            ("Air Trapping",  "--",  STAT_PURPLE, "stat_trap"),
        ]

        self.stat_widgets = {}
        for label, value, color, key in stat_defs:
            card, val_lbl = make_stat_card(label, value, color)
            stats_grid.addWidget(card)
            self.stat_widgets[key] = val_lbl

        self.main_layout.addWidget(stats_outer)

        # ═══════ Viewer ═══════
        self.viewer = SliceViewer(self)
        self.main_layout.addWidget(self.viewer, 1)

        self.setLayout(self.main_layout)

    # ── Slots ──

    def _on_mesh_changed(self, level):
        mesh = self.mesh_presets[level]
        nx, ny, nz = mesh
        sx = getattr(self, '_mesh_sp_x', 0.574)
        sy = getattr(self, '_mesh_sp_y', 0.574)
        sz = getattr(self, '_mesh_sp_z', 5.0)
        mx = getattr(self, '_mesh_mx', 512)
        mz = getattr(self, '_mesh_mz', 46)
        self.mesh_value_label.label().setText(f"{level}  [{nx}x{ny}x{nz}]")
        self.mesh_spacing_label.label().setText(
            f"XY ~{mx*sx/nx:.1f}mm  Z ~{mz*sz/nz:.1f}mm  ({nx*ny*nz} ctrl pts)")

    def _update_mesh_from_insp(self):
        try:
            d = self.insp_edit.line_edit().text().strip()
            if not d or not os.path.isdir(d): return
            meta = get_dcm_meta(d)
        except Exception:
            return
        try:
            dcm_f = glob.glob(os.path.join(d, '*.dcm')) or [
                os.path.join(d, f) for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))]
            ds = pydicom.dcmread(dcm_f[0], force=True)
            rows = int(getattr(ds, 'Rows', 512))
            cols = int(getattr(ds, 'Columns', 512))
            ps = getattr(ds, 'PixelSpacing', [0.574, 0.574])
            sp_y, sp_x = float(ps[0]), float(ps[1])
            sp_z = float(getattr(ds, 'SliceThickness', 5.0))
            self._mesh_mx, self._mesh_mz = cols, meta['num_slices']
            self._mesh_sp_x, self._mesh_sp_y, self._mesh_sp_z = sp_x, sp_y, sp_z
            self.mesh_presets = build_mesh_presets(cols, rows, meta['num_slices'], sp_x, sp_y, sp_z)
            self._on_mesh_changed(self.mesh_slider.value())
        except Exception:
            pass

    def _browse(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if d:
            edit.line_edit().setText(d)
            if edit is self.insp_edit:
                self._update_mesh_from_insp()

    def _start_compute(self):
        insp = self.insp_edit.line_edit().text().strip()
        exp  = self.exp_edit.line_edit().text().strip()
        if not insp or not exp:
            QMessageBox.warning(self, "Warning", "Please select both inspiratory and expiratory directories")
            return
        if not os.path.isdir(insp):
            QMessageBox.warning(self, "Warning", f"Invalid path: {insp}"); return
        if not os.path.isdir(exp):
            QMessageBox.warning(self, "Warning", f"Invalid path: {exp}"); return
        try:
            im = get_dcm_meta(insp); em = get_dcm_meta(exp)
        except Exception as e:
            QMessageBox.warning(self, "DICOM Read Error", str(e)); return
        mismatches = []
        if im['patient_id'] != em['patient_id']:
            mismatches.append(f"Patient ID: {im['patient_id']} vs {em['patient_id']}")
        try: si, se = float(im['slice_thickness']), float(em['slice_thickness'])
        except: si = se = 0
        if si > 0 and se > 0 and abs(si - se) > 0.01:
            mismatches.append(f"Slice thickness: {si} vs {se} mm")
        if im['num_slices'] != em['num_slices']:
            mismatches.append(f"Slice count: {im['num_slices']} vs {em['num_slices']}")
        if mismatches:
            QMessageBox.warning(self, "DICOM Validation Failed",
                                "Mismatches:\n" + "\n".join(mismatches)); return

        self.calc_btn.button().setEnabled(False)
        self.mesh_slider.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.label().setText("Computing ...")
        self.worker = ComputeWorker(insp, exp, mesh=self.mesh_presets[self.mesh_slider.value()])
        self.worker.progress.connect(lambda m: self.progress_label.label().setText(m))
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, data):
        self.progress_bar.setVisible(False)
        self.progress_label.label().setText("")
        self.calc_btn.button().setEnabled(True)
        self.mesh_slider.setEnabled(True)
        self.viewer.set_data(data)
        mx, my, mz = data['mesh']
        self.colorbar_info.label().setText(
            f"Mesh [{mx}x{my}x{mz}]  |  "
            f"{data['vmin']:.1f}%  <  100 (no change)  <  {data['vmax']:.1f}%")

        self.stat_widgets['stat_lung'].setText(f"{data['lung_vol_ml']:.0f} mL")
        self.stat_widgets['stat_mean'].setText(f"{data['mean_vc']:.2f}%")
        self.stat_widgets['stat_sd'].setText(f"{data['std_vc']:.2f}%")
        self.stat_widgets['stat_p5'].setText(f"{data['p5_vc']:.2f}%")
        self.stat_widgets['stat_p95'].setText(f"{data['p95_vc']:.2f}%")
        self.stat_widgets['stat_low'].setText(f"{data['low_vent_pct']:.1f}%")
        self.stat_widgets['stat_high'].setText(f"{data['high_vent_pct']:.1f}%")
        self.stat_widgets['stat_mismatch'].setText(f"{data['mismatch_ml']:.1f} mL")
        self.stat_widgets['stat_trap'].setText(f"{data['trap_ml']:.1f} mL")

        btn = self.view_btns.checkedButton()
        if btn:
            self.viewer.switch_axis(self.AXIS_MAP[btn.text()])

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.progress_label.label().setText("")
        self.calc_btn.button().setEnabled(True)
        self.mesh_slider.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Computation failed:\n{msg}")

    def _on_axis_changed(self):
        if not hasattr(self, 'viewer') or self.viewer.data is None: return
        btn = self.view_btns.checkedButton()
        if btn:
            self.viewer.switch_axis(self.AXIS_MAP[btn.text()])

    def _on_overlay_toggled(self, checked):
        if hasattr(self, 'viewer'):
            self.viewer.toggle_overlay(checked)

    def _export_image(self):
        if not hasattr(self, 'viewer') or self.viewer.data is None:
            QMessageBox.information(self, "Info", "Please complete computation before exporting")
            return
        pixmap = self.viewer.image_label.pixmap()
        if pixmap is None:
            QMessageBox.warning(self, "Info", "No image to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "lung_heatmap.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            pixmap.save(path)
            self.progress_label.label().setText(f"Exported: {os.path.basename(path)}")


if __name__ == '__main__':
    set_color_theme("blue")
    set_appearance_mode("dark")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()
