"""Interactive CNN teaching lab. Run: streamlit run cnn_teaching_lab.py."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

from image_processing import (
    ACTIVATIONS, KERNELS, PLAYGROUND_OPERATIONS, POOLING, PREPROCESSING, Layer, Stage,
    apply_operation, default_network, make_samples, run_network,
)
from operation_view import operation_html


BLOCK_STYLE = """
.sortable-component { padding: 0; }
.sortable-container { background: transparent; }
.sortable-item, .sortable-item:hover {
    background: #dceeff; color: #123556; border: 2px solid #78b4e8;
    border-left: 9px solid #2979c1; border-radius: 8px;
    padding: 14px; margin: 8px 0; font-weight: 600; cursor: grab;
}
"""


def draw_grid(
    a: np.ndarray, title: str, show_values: bool, cell_size: int,
    rescale: bool = False, overlay: tuple[int, int, int, int] | None = None,
    original_bounds: tuple[int, int, int, int] | None = None,
):
    st.markdown(f"**{title}**")
    if a.size == 0:
        st.warning("Empty grid: use a smaller window, stride 1, or SAME padding.")
        return
    lo, hi = float(a.min()), float(a.max())
    shown = a.astype(float)
    if rescale and hi > lo:
        shown = (shown - lo) / (hi - lo) * 255
    elif rescale:
        # Constant maps must not disappear as black after rescaling.
        shown = np.full(a.shape, np.clip(lo, 0, 255))
    shown = np.clip(shown, 0, 255)
    rows, cols = a.shape
    fig, ax = plt.subplots(figsize=(
        max(3, min(10, cols * cell_size / 100)),
        max(3, min(10, rows * cell_size / 100)),
    ))
    ax.imshow(shown, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if original_bounds is not None:
        top, left, height, width = original_bounds
        tint = np.zeros((*a.shape, 4))
        tint[:] = (1, .65, .18, .55)
        tint[top:top + height, left:left + width, 3] = 0
        ax.imshow(tint, interpolation="nearest")
        ax.add_patch(plt.Rectangle(
            (left - .5, top - .5), width, height, fill=False,
            edgecolor="#ffa62d", linewidth=2, linestyle="--",
        ))
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#6e91ad", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if show_values and max(rows, cols) <= 24:
        for r in range(rows):
            for c in range(cols):
                value = float(a[r, c])
                label = str(int(value)) if value.is_integer() else f"{value:.2g}"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=max(4, min(10, cell_size // 7)),
                        color="#151515" if shown[r, c] > 140 else "#ffffff")
    if overlay is not None:
        y, x, h, w = overlay
        ax.add_patch(plt.Rectangle((x - .5, y - .5), w, h, fill=False, edgecolor="#ff374f", linewidth=3))
    fig.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"{rows} × {cols} pixels · actual range {lo:.4g} to {hi:.4g}")
    if show_values and max(rows, cols) > 24:
        st.caption("Large grid: values are in the table below to keep the image readable.")
        with st.expander(f"{title} · numeric pixel table"):
            st.dataframe(a, use_container_width=True)
    if rescale and hi > lo:
        st.caption("Display rescaled to 0–255; calculations and printed values are unchanged.")
    elif lo < 0 or hi > 255:
        st.caption("Display clipped to 0–255; calculations and printed values are unchanged.")


def inspect_kernel(stage: Stage, show_values: bool, cell_size: int, rescale: bool, key: str):
    if stage.convolution.size == 0:
        st.info("This kernel has no valid positions. Adjust padding or grid size.")
        return
    total = stage.convolution.size
    controls, preview = st.columns([1, 2])
    with controls:
        step = st.slider("Kernel position", 0, total - 1, key=f"{key}_step") if total > 1 else 0
        play = st.button("Play kernel scan", key=f"{key}_play")
        st.caption("Convolution here uses the CNN convention: cross-correlation (no kernel flip).")
        st.dataframe(KERNELS[stage.layer.kernel], use_container_width=True)
        row, col = divmod(step, stage.convolution.shape[1])
        y, x = row * stage.layer.stride, col * stage.layer.stride
        patch = stage.padded[y:y + 3, x:x + 3]
        st.write("Pixel × weight")
        st.dataframe(patch * KERNELS[stage.layer.kernel], use_container_width=True)
        st.info(f"Sum = {stage.convolution[row, col]:.4g} at output ({row}, {col}).")
    with preview:
        frame = st.empty()
        with frame.container():
            draw_grid(stage.padded, "Padded input / kernel window", show_values, cell_size,
                      rescale, (y, x, 3, 3))
        if play:
            positions = np.unique(np.linspace(0, total - 1, min(total, 48), dtype=int))
            st.caption("Previewing up to 48 positions; use the slider to inspect every position.")
            for position in positions:
                row, col = divmod(int(position), stage.convolution.shape[1])
                with frame.container():
                    draw_grid(stage.padded, f"Kernel scan: output ({row}, {col})",
                              show_values, cell_size, rescale,
                              (row * stage.layer.stride, col * stage.layer.stride, 3, 3))
                time.sleep(0.06)


def playground(image: np.ndarray, show_values: bool, cell_size: int, rescale: bool):
    st.subheader("Image-processing playground")
    st.write("Try one operation at a time. Compare the output shape and pixel values with the original.")
    operation = st.selectbox("Operation", list(PLAYGROUND_OPERATIONS))
    st.info(PLAYGROUND_OPERATIONS[operation])
    size, stride, factor, padding = 2, 1, 2, "valid"
    windowed = operation == "Gaussian smoothing" or operation in POOLING
    if windowed:
        c1, c2 = st.columns(2)
        if operation == "Gaussian smoothing":
            c1.caption("Kernel: 3 × 3 Gaussian weighted-average window.")
            stride = c2.selectbox("Stride", [1, 2, 3], key="lab_stride")
        else:
            size = c1.selectbox("Pooling window", [2, 3, 4], key="lab_pool_size")
            stride = c2.selectbox("Pooling stride", [1, 2, 3, 4], index=1, key="lab_pool_stride")
        use_padding = st.toggle("Padding", value=True, key="lab_use_padding",
                                help="On: SAME zero padding. Off: VALID, with no added border.")
        padding = "same" if use_padding else "valid"
    elif operation == "Downsampling (nearest neighbor)":
        factor = st.select_slider("Downsampling factor", [2, 3, 4], value=2)
    else:
        st.caption("This is a per-pixel operation: a 1 × 1 cursor replaces the moving kernel. "
                   "Padding is not needed.")
    result = apply_operation(image, operation, size=size, stride=stride, padding=padding, factor=factor)
    top, left, height, width = result.original_bounds
    bottom = result.incoming.shape[0] - top - height
    right = result.incoming.shape[1] - left - width
    has_padding = any((top, bottom, left, right))
    if windowed:
        if has_padding:
            st.caption(f"Zero padding: top {top}, bottom {bottom}, left {left}, right {right}. "
                       "Orange cells are added zeros; the dashed border encloses the original image.")
            if operation in POOLING:
                st.caption("Padded zeros participate in pooling: they can become the minimum "
                           "and count in the average.")
        elif padding == "same":
            st.caption("Padding is on (SAME), but this window/stride needs no extra border cells.")
        else:
            st.caption("Padding is off (VALID): only windows entirely inside the original image are used.")
    st.markdown("**Kernel / window:** the neighborhood used for one output pixel. "
                "**Stride:** how many input cells the window moves between output pixels. "
                "**Padding:** extra border cells (zeros here), allowing windows to reach image edges.")
    with st.expander("Definitions of the operations"):
        for name, definition in PLAYGROUND_OPERATIONS.items():
            st.markdown(f"**{name}:** {definition}")
    if result.kernel is not None:
        with st.expander("Gaussian kernel weights"):
            st.dataframe(result.kernel, use_container_width=True)
    if st.toggle("View operation", key="lab_view_operation",
                 help="Watch the window move and construct every output pixel. Pause or restart at any time."):
        st.caption("Use Play, Pause, Restart, Next pixel, or scrub the Built pixels slider. "
                   "Changing the image or operation resets playback.")
        components.html(operation_html(result, show_values, rescale), height=850, scrolling=True)
    else:
        before, after = st.columns(2)
        with before:
            draw_grid(result.incoming, "Input · padded matrix" if has_padding else "Input · 8-bit grayscale",
                      show_values, cell_size, original_bounds=result.original_bounds if has_padding else None)
        with after:
            draw_grid(result.output, operation, show_values, cell_size, rescale)


def change_network(action: str, template: str = "Edge detector"):
    layers = st.session_state.layers
    if action == "default":
        st.session_state.layers = default_network()
    elif action == "clear":
        st.session_state.layers = []
    elif action == "add":
        templates = {
            "Edge detector": Layer(),
            "Gaussian smoothing": Layer(name="Gaussian smoothing", kernel="Gaussian blur", activation="None"),
            "Max pooling": Layer(name="Max pooling", kernel="Identity", activation="None", pooling="Max pooling"),
            "Average pooling": Layer(name="Average pooling", kernel="Identity", activation="None", pooling="Average pooling"),
            "Custom convolution": Layer(name="Custom convolution", kernel="Identity", activation="None"),
        }
        layers.append(templates[template])
    else:
        raise ValueError(f"Unknown network action: {action}")
    st.session_state.board_revision += 1


def edit_layer(layer: Layer):
    prefix = layer.id
    c1, c2 = st.columns(2)
    layer.name = c1.text_input("Block name", layer.name, key=f"{prefix}_name")
    layer.preprocess = c2.selectbox("Preprocessing", PREPROCESSING,
                                   index=PREPROCESSING.index(layer.preprocess), key=f"{prefix}_pre")
    c1, c2, c3 = st.columns(3)
    layer.kernel = c1.selectbox("Convolution kernel", list(KERNELS),
                               index=list(KERNELS).index(layer.kernel), key=f"{prefix}_kernel")
    layer.padding = c2.selectbox("Padding", ["same", "valid"],
                                index=["same", "valid"].index(layer.padding), key=f"{prefix}_pad")
    layer.stride = c3.selectbox("Convolution stride", [1, 2, 3],
                              index=layer.stride - 1, key=f"{prefix}_stride")
    c1, c2 = st.columns(2)
    layer.activation = c1.selectbox("Activation", ACTIVATIONS,
                                   index=ACTIVATIONS.index(layer.activation), key=f"{prefix}_act")
    layer.pooling = c2.selectbox("Pooling", POOLING, index=POOLING.index(layer.pooling),
                               key=f"{prefix}_pool")
    if layer.pooling != "None":
        c1, c2 = st.columns(2)
        layer.pool_size = c1.selectbox("Pooling window", [2, 3, 4],
                                      index=layer.pool_size - 2, key=f"{prefix}_size")
        layer.pool_stride = c2.selectbox("Pooling stride", [1, 2, 3, 4],
                                        index=layer.pool_stride - 1, key=f"{prefix}_pstride")


def network_builder(image: np.ndarray, show_values: bool, cell_size: int, rescale: bool):
    st.subheader("Build a CNN feature extractor")
    st.caption("Fixed filters, not a trained classifier: no learned weights, training, or predictions. "
               "Each block runs preprocessing → convolution → activation → optional pooling.")
    c1, c2 = st.columns(2)
    c1.button("Restore 10-layer example", on_click=change_network, args=("default",),
              use_container_width=True)
    c2.button("Start empty", on_click=change_network, args=("clear",), use_container_width=True)
    board, editor = st.columns([1, 2])
    with board:
        st.markdown("#### Block palette")
        template = st.selectbox("Block type", [
            "Edge detector", "Gaussian smoothing", "Max pooling",
            "Average pooling", "Custom convolution",
        ])
        st.button("＋ Add block", on_click=change_network, args=("add", template),
                  use_container_width=True)
        st.markdown("#### Network · top to bottom")
        st.caption("Add blocks from the palette, then drag them into order. Changes run immediately.")
        layers = st.session_state.layers
        labels = {
            f"{index + 1:02d} · {layer.name or 'Untitled'}": layer
            for index, layer in enumerate(layers)
        }
        if layers:
            ordered = sort_items(list(labels), direction="vertical", custom_style=BLOCK_STYLE,
                                 key=f"board_{st.session_state.board_revision}")
            reordered = [labels[label] for label in ordered]
            if [layer.id for layer in reordered] != [layer.id for layer in layers]:
                st.session_state.layers = reordered
                st.session_state.board_revision += 1
                st.rerun()
        else:
            st.info("Your network is empty. Add a block to begin.")
    with editor:
        if layers:
            ids = [layer.id for layer in layers]
            selected = st.selectbox(
                "Configure block", ids, key="selected_block",
                format_func=lambda uid: f"{ids.index(uid) + 1}. {next(x.name for x in layers if x.id == uid)}",
            )
            layer = next(layer for layer in layers if layer.id == selected)
            previous_name = layer.name
            edit_layer(layer)
            if layer.name != previous_name:
                st.session_state.board_revision += 1
                st.rerun()
            duplicate, remove = st.columns(2)
            if duplicate.button("Duplicate block"):
                copy = replace(layer, id=Layer().id, name=f"{layer.name} copy")
                layers.insert(layers.index(layer) + 1, copy)
                st.session_state.board_revision += 1
                st.rerun()
            if remove.button("Remove block"):
                layers.remove(layer)
                st.session_state.board_revision += 1
                st.rerun()
            st.caption("SAME uses zero padding. Pooling uses no padding. "
                       "Pixel-based preprocessing (e.g. invert at 255) also acts on raw feature values.")
        else:
            st.write("The input passes through unchanged until you add a layer.")

    stages = run_network(image, layers)
    output = stages[-1].output if stages else image
    st.divider()
    st.markdown("#### Network results")
    st.dataframe([
        {
            "Layer": index + 1, "Block": stage.layer.name,
            "Kernel": stage.layer.kernel, "Activation": stage.layer.activation,
            "Pooling": stage.layer.pooling,
            "Input": f"{stage.incoming.shape[0]} × {stage.incoming.shape[1]}",
            "Output": f"{stage.output.shape[0]} × {stage.output.shape[1]}",
        }
        for index, stage in enumerate(stages)
    ], use_container_width=True, hide_index=True)
    if stages and output.size == 0:
        st.error(f"Stopped at block {len(stages)} ({stages[-1].layer.name}): the grid became empty. "
                 "Use SAME padding, a smaller pooling window/stride, or a larger input grid. "
                 "Later blocks have not run.")
    c1, c2 = st.columns(2)
    with c1:
        draw_grid(image, "Network input · 8-bit grayscale", show_values, cell_size)
    with c2:
        draw_grid(output, f"Network output · {len(stages)} blocks applied",
                  show_values, cell_size, rescale)
    if stages:
        selected_stage = st.selectbox(
            "Inspect intermediate layer", range(len(stages)),
            format_func=lambda index: f"{index + 1}. {stages[index].layer.name}",
        )
        stage = stages[selected_stage]
        views = {
            "Input": stage.incoming, "Preprocessing": stage.preprocessed,
            "Convolution": stage.convolution, "Activation": stage.activated, "Pooling / output": stage.output,
        }
        selected_view = st.radio("Processing step", list(views), index=4, horizontal=True)
        draw_grid(views[selected_view], selected_view, show_values, cell_size,
                  rescale and not (selected_stage == 0 and selected_view == "Input"))
        with st.expander("Inspect / animate this convolution"):
            inspect_kernel(stage, show_values, cell_size, rescale, f"network_{stage.layer.id}")
    st.download_button(
        "Download lesson configuration",
        data=json.dumps({
            "image": st.session_state.image_choice, "grid_size": image.shape[0],
            "layers": [asdict(layer) for layer in layers],
        }, indent=2),
        file_name="cnn_lesson_config.json", mime="application/json",
    )


def main():
    st.set_page_config(page_title="CNN Teaching Lab", page_icon="▦", layout="wide")
    st.title("CNN Teaching Lab")
    st.caption("Generate pixels. Explore filters. Build a feature extractor.")
    if "layers" not in st.session_state:
        st.session_state.layers = default_network()
        st.session_state.board_revision = 0
    with st.sidebar:
        st.header("Image generator")
        grid_size = st.slider("Image grid size", 8, 64, 16, step=8,
                              help="Number of rows and columns, not display zoom.")
        samples = make_samples(grid_size)
        choice = st.selectbox("Image pattern", list(samples), key="image_choice")
        st.caption("Generated as unsigned 8-bit integers: 0 = black, 255 = white. "
                   "Patterns scale with the chosen grid size; noise is reproducible.")
        st.divider()
        st.subheader("Grid display")
        show_values = st.checkbox("Show pixel values", value=grid_size <= 16)
        cell_size = st.slider("Grid zoom", 25, 105, 52)
        rescale = st.toggle("Rescale feature-map contrast", value=False,
                            help="Display only. Off: clip brightness to 0–255. "
                                 "On: stretch each nonconstant feature map to 0–255.")
        st.info("Inputs are 8-bit. Feature maps use floating-point numbers so negative edges, "
                "averages, and values above 255 are not lost.")
    playground_tab, network_tab = st.tabs(["Image processing", "CNN network builder"])
    with playground_tab:
        playground(samples[choice], show_values, cell_size, rescale)
    with network_tab:
        network_builder(samples[choice], show_values, cell_size, rescale)


if __name__ == "__main__":
    main()
