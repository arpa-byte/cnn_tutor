# CNN Teaching Lab

A Streamlit classroom lab with two workspaces:

- **Image processing:** downsampling (nearest neighbor), max/min/average pooling, Gaussian smoothing, min–max normalization to 0–1, and inversion. Dashboard definitions explain each operation, kernels/windows, stride, and padding. Toggle zero padding for smoothing; added border cells are highlighted orange in the input matrix. SAME padding adds only the cells needed for the chosen window and stride. Pooling uses no padding.
- **CNN network builder:** start with a ten-layer feature extractor or an empty network. Add blocks from the palette, click a block to configure it, and use its arrows to reorder. Duplicate or remove the selected block. Watch preprocessing, moving convolution kernels, activation, and pooling build each output pixel.

The same ten generated image patterns support 8 × 8 through 64 × 64 grids. Input pixels are unsigned 8-bit integers (0–255). Feature maps retain floating-point values, including negative values and values above 255. Display contrast rescaling is optional and never changes calculations. By default, brightness is clipped to 0–255; numeric labels show actual values.

Blocks use fixed, editable filter choices; this is a demonstration of CNN feature extraction, **not training or classification**. Each block contains preprocessing, convolution, activation, and optional pooling. Pooling-only palette blocks use identity convolution with no activation. If a custom network shrinks its grid to nothing, execution stops at that block with an explanation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run cnn_teaching_lab.py
```

The browser opens at the local URL printed by Streamlit. The image controls are shared across both tabs. Download the image settings and current architecture as JSON from the network builder.

The playground shows the operation player by default: see the moving kernel/window and output built one pixel at a time, with the current calculation. Play, pause, restart, step forward, scrub, or change speed directly in the animation. All output positions are included; changing the input or operation resets playback. Per-pixel operations use a 1 × 1 cursor rather than a neighborhood kernel. Enable **Show only output** to replace the player with static input and output grids.

Network playback also opens by default. **Entire network** follows all operations in order, moving each completed output to the input side for the next operation or layer. **One layer** plays only the chosen layer. Click a numbered layer inside the player to restart there; earlier layers are already calculated when jumping ahead. Use **Pause**, **Continue**, **Restart from beginning**, or **Restart selected layer**, and adjust speed or scrub the current operation. Editing the network or image resets playback. **Show only output** switches to static network input and final output; an expander contains the selected block's intermediate values and kernel weights.

Numerical operations live in `image_processing.py`, separately from the Streamlit UI so they can be reused in a future frontend.

```bash
python -m unittest discover -s tests
```

The playback control tests also require Node.js to execute the embedded JavaScript.
