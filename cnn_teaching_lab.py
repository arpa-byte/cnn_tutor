"""Interactive CNN teaching lab. Run: streamlit run cnn_teaching_lab.py."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from image_processing import (
    ACTIVATIONS, KERNELS, PATTERN_FILTER, PLAYGROUND_OPERATIONS, POOLING, PREPROCESSING, Layer,
    apply_operation, default_network, make_samples, pattern_input, run_network,
)
from operation_view import HEATMAP_COLORS, network_html, operation_html


def draw_grid(
    a: np.ndarray, title: str, show_values: bool, cell_size: int,
    rescale: bool = False, overlay: tuple[int, int, int, int] | None = None,
    original_bounds: tuple[int, int, int, int] | None = None,
    heatmap: bool = False, bipolar: bool = False,
):
    st.markdown(f"**{title}**")
    if a.size == 0:
        st.warning("Empty grid: use a smaller window, stride 1, or SAME padding.")
        return
    lo, hi = float(a.min()), float(a.max())
    shown = a.astype(float)
    heat_limit = max(abs(lo), abs(hi), 1)
    if heatmap:
        palette = LinearSegmentedColormap.from_list(
            "pattern_response", np.array(HEATMAP_COLORS) / 255,
        )
        colors = palette((a + heat_limit) / (2 * heat_limit))
        shown = (colors[..., :3] @ np.array([.2126, .7152, .0722])) * 255
    elif bipolar:
        shown = (shown + 1) * 127.5
    elif rescale and hi > lo:
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
    if heatmap:
        plot = ax.imshow(a, cmap=palette, vmin=-heat_limit, vmax=heat_limit, interpolation="nearest")
        fig.colorbar(plot, ax=ax, shrink=.8, label="Pattern response")
    else:
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
    if heatmap:
        st.caption("Blue = negative response · white = zero · red = positive response. "
                   "Colors do not change the numeric results.")
    elif bipolar:
        st.caption("Binary pattern: −1 = black, +1 = white.")
    elif rescale and hi > lo:
        st.caption("Display rescaled to 0–255; calculations and printed values are unchanged.")
    elif lo < 0 or hi > 255:
        st.caption("Display clipped to 0–255; calculations and printed values are unchanged.")


def randomize_pattern_filter():
    weights = np.random.default_rng().integers(-1, 2, (3, 3))
    for row, col in np.ndindex(3, 3):
        st.session_state[f"pattern_weight_{row}_{col}"] = float(weights[row, col])


def pattern_controls(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, str, bool]:
    if "pattern_run" not in st.session_state:
        st.session_state.pattern_run = {
            "kernel": PATTERN_FILTER.copy(), "stride": 1, "padding": "valid",
            "average": True, "input": "Slide-style X (9 × 9)",
        }
    settings = st.session_state.pattern_run
    st.markdown("#### Convolution for image pattern recognition")
    st.caption("Inspired by slide 88: a −1/+1 input, a 3 × 3 filter, and a response heatmap. "
               "Edit or randomize the weights, then click Run convolution to apply them.")
    st.button("Randomize 3 × 3 filter", on_click=randomize_pattern_filter,
              key="pattern_randomize",
              help="Fill the editor with random −1, 0, or +1 weights. Click Run convolution to apply.")
    with st.form("pattern_filter_form"):
        input_choice = st.selectbox(
            "Pattern input", ["Slide-style X (9 × 9)", "Sidebar image (threshold at 128)"],
            index=0 if settings["input"] == "Slide-style X (9 × 9)" else 1,
        )
        st.markdown("**Filter parameters · 3 × 3**")
        weights = np.empty((3, 3))
        for row in range(3):
            cells = st.columns(3)
            for col, cell in enumerate(cells):
                key = f"pattern_weight_{row}_{col}"
                if key not in st.session_state:
                    st.session_state[key] = float(settings["kernel"][row, col])
                weights[row, col] = cell.number_input(
                    f"Weight ({row + 1}, {col + 1})", step=.25, format="%.2f", key=key,
                )
        c1, c2 = st.columns(2)
        stride = c1.selectbox("Stride", [1, 2, 3], index=settings["stride"] - 1,
                             key="pattern_stride")
        padded = c2.toggle("Padding", value=settings["padding"] == "same", key="pattern_padding",
                           help="On: SAME zero padding. Off: VALID, as in the slide example.")
        average = st.checkbox(
            "Average the 9 products", value=settings["average"], key="pattern_average",
            help="Divide the dot product by 9. Uncheck to see the raw convolution sum.",
        )
        if st.form_submit_button("Run convolution"):
            if not np.isfinite(weights).all():
                st.error("Filter weights must be finite numbers. The previous result is unchanged.")
            else:
                st.session_state.pattern_run = {
                    "kernel": weights, "stride": stride, "padding": "same" if padded else "valid",
                    "average": average, "input": input_choice,
                }
                st.session_state.pattern_revision = st.session_state.get("pattern_revision", 0) + 1
    settings = st.session_state.pattern_run
    source = pattern_input(None if settings["input"] == "Slide-style X (9 × 9)" else image)
    st.caption("Heatmap values use the last applied filter. With −1/+1 pixels and weights, "
               "averaging gives +1 for a perfect match and −1 for the opposite pattern. "
               "Other weights can produce values outside this range. The kernel is not flipped.")
    return source, settings["kernel"], settings["stride"], settings["padding"], settings["average"]


def playground(image: np.ndarray, show_values: bool, cell_size: int, rescale: bool):
    st.subheader("Image-processing playground")
    st.write("Try one operation at a time. Compare the output shape and pixel values with the original.")
    operation = st.selectbox("Operation", list(PLAYGROUND_OPERATIONS))
    st.info(PLAYGROUND_OPERATIONS[operation])
    size, stride, factor, padding = 2, 1, 2, "valid"
    kernel, average = None, True
    windowed = operation in ("Gaussian smoothing", "Convolution for pattern recognition") or operation in POOLING
    if operation == "Convolution for pattern recognition":
        image, kernel, stride, padding, average = pattern_controls(image)
    elif windowed:
        c1, c2 = st.columns(2)
        if operation == "Gaussian smoothing":
            c1.caption("Kernel: 3 × 3 Gaussian weighted-average window.")
            stride = c2.selectbox("Stride", [1, 2, 3], key="lab_stride")
            use_padding = st.toggle("Padding", value=True, key="lab_use_padding",
                                    help="On: SAME zero padding. Off: VALID, with no added border.")
            padding = "same" if use_padding else "valid"
        else:
            size = c1.selectbox("Pooling window", [2, 3, 4], key="lab_pool_size")
            stride = c2.selectbox("Pooling stride", [1, 2, 3, 4], index=1, key="lab_pool_stride")
    elif operation == "Downsampling (nearest neighbor)":
        factor = st.select_slider("Downsampling factor", [2, 3, 4], value=2)
    else:
        st.caption("This is a per-pixel operation: a 1 × 1 cursor replaces the moving kernel. "
                   "Padding is not needed.")
    result = apply_operation(image, operation, size=size, stride=stride, padding=padding, factor=factor,
                             kernel=kernel, average_products=average)
    top, left, height, width = result.original_bounds
    bottom = result.incoming.shape[0] - top - height
    right = result.incoming.shape[1] - left - width
    has_padding = any((top, bottom, left, right))
    if windowed:
        if has_padding:
            st.caption(f"Zero padding: top {top}, bottom {bottom}, left {left}, right {right}. "
                       "Orange cells are added zeros; the dashed border encloses the original image.")
        elif padding == "same":
            st.caption("Padding is on (SAME), but this window/stride needs no extra border cells.")
        else:
            st.caption("No padding (VALID): only windows entirely inside the original image are used.")
    st.markdown("**Kernel / window:** the neighborhood used for one output pixel. "
                "**Stride:** how many input cells the window moves between output pixels. "
                "**Padding:** extra border cells (zeros here), allowing windows to reach image edges.")
    with st.expander("Definitions of the operations"):
        for name, definition in PLAYGROUND_OPERATIONS.items():
            st.markdown(f"**{name}:** {definition}")
    if result.kernel is not None:
        with st.expander("Applied filter weights" if result.heatmap else "Gaussian kernel weights"):
            st.dataframe(result.kernel, use_container_width=True)
    if not st.toggle("Show only output", value=False, key="lab_show_only_output",
                     help="Hide the operation player and show only the input and output grids."):
        st.caption("Use Play, Pause, Restart, Next pixel, or scrub the Built pixels slider. "
                   "Changing the image or operation resets playback.")
        run_id = st.session_state.get("pattern_revision", 0) if result.heatmap else None
        components.html(operation_html(result, show_values, rescale, run_id), height=850, scrolling=True)
    else:
        if result.heatmap:
            before, filter_column, after = st.columns([1, .45, 1])
            with filter_column:
                st.markdown("**Applied filter**")
                st.dataframe(result.kernel, use_container_width=True)
                st.caption(f"Sum of products ÷ {result.divisor:g}")
        else:
            before, after = st.columns(2)
        with before:
            title = "Input · −1/+1 pattern" if result.bipolar_input else "Input · 8-bit grayscale"
            draw_grid(result.incoming, f"{title} · padded" if has_padding else title,
                      show_values, cell_size, original_bounds=result.original_bounds if has_padding else None,
                      bipolar=result.bipolar_input)
        with after:
            draw_grid(result.output, "Output heatmap" if result.heatmap else operation,
                      show_values, cell_size, rescale, heatmap=result.heatmap)


def change_network(action: str, template: str = "Edge detector"):
    layers = st.session_state.layers
    if action == "default":
        st.session_state.layers = default_network()
        st.session_state.selected_block = st.session_state.layers[0].id
    elif action == "clear":
        st.session_state.layers = []
        st.session_state.selected_block = None
    elif action == "add":
        templates = {
            "Edge detector": Layer(),
            "Gaussian smoothing": Layer(name="Gaussian smoothing", kernel="Gaussian blur", activation="None"),
            "Max pooling": Layer(name="Max pooling", kernel="Identity", activation="None", pooling="Max pooling"),
            "Average pooling": Layer(name="Average pooling", kernel="Identity", activation="None", pooling="Average pooling"),
            "Custom convolution": Layer(name="Custom convolution", kernel="Identity", activation="None"),
        }
        layers.append(templates[template])
        st.session_state.selected_block = layers[-1].id
    else:
        raise ValueError(f"Unknown network action: {action}")


def select_layer(layer_id: str):
    st.session_state.selected_block = layer_id


def move_layer(layer_id: str, offset: int):
    layers = st.session_state.layers
    index = next(index for index, layer in enumerate(layers) if layer.id == layer_id)
    destination = index + offset
    if 0 <= destination < len(layers):
        layers.insert(destination, layers.pop(index))
    st.session_state.selected_block = layer_id


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
    layers = st.session_state.layers
    ids = [layer.id for layer in layers]
    if st.session_state.get("selected_block") not in ids:
        st.session_state.selected_block = ids[0] if ids else None
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
        st.caption("Click a block to edit it. Use the arrows to reorder. Changes run immediately.")
        if layers:
            for index, layer in enumerate(layers):
                block, up, down = st.columns([6, 1, 1])
                block.button(
                    f"{index + 1:02d} · {layer.name or 'Untitled'}",
                    key=f"select_{layer.id}", on_click=select_layer, args=(layer.id,),
                    type="primary" if layer.id == st.session_state.selected_block else "secondary",
                    use_container_width=True,
                )
                up.button("↑", key=f"up_{layer.id}", on_click=move_layer, args=(layer.id, -1),
                          disabled=index == 0, help="Move block up")
                down.button("↓", key=f"down_{layer.id}", on_click=move_layer, args=(layer.id, 1),
                            disabled=index == len(layers) - 1, help="Move block down")
        else:
            st.info("Your network is empty. Add a block to begin.")
    with editor:
        if layers:
            layer = next(layer for layer in layers if layer.id == st.session_state.selected_block)
            st.markdown(f"#### Configure block {layers.index(layer) + 1}")
            previous_name = layer.name
            edit_layer(layer)
            if layer.name != previous_name:
                st.rerun()
            duplicate, remove = st.columns(2)
            if duplicate.button("Duplicate block"):
                copy = replace(layer, id=Layer().id, name=f"{layer.name} copy")
                layers.insert(layers.index(layer) + 1, copy)
                st.session_state.selected_block = copy.id
                st.rerun()
            if remove.button("Remove block"):
                index = layers.index(layer)
                layers.remove(layer)
                st.session_state.selected_block = layers[min(index, len(layers) - 1)].id if layers else None
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
    show_only_output = st.toggle(
        "Show only output", key="network_show_only_output",
        help="Replace network playback with static input and final output grids.",
    )
    if stages and not show_only_output:
        st.caption("Play the entire network or one layer. Each completed output becomes the next "
                   "operation's input. Click a layer in the player to restart there.")
        components.html(
            network_html(stages, show_values, rescale, st.session_state.selected_block),
            height=1050, scrolling=True,
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            draw_grid(image, "Network input · 8-bit grayscale", show_values, cell_size)
        with c2:
            draw_grid(output, f"Network output · {len(stages)} blocks applied",
                      show_values, cell_size, rescale)
    if stages:
        stage = next((stage for stage in stages if stage.layer.id == st.session_state.selected_block), None)
        if stage is None:
            st.info("This block was not reached because an earlier block produced an empty grid.")
        else:
            with st.expander("Selected block · static processing steps and kernel"):
                views = {
                    "Input": stage.incoming, "Preprocessing": stage.preprocessed,
                    "Convolution": stage.convolution, "Activation": stage.activated,
                    "Pooling / output": stage.output,
                }
                selected_view = st.radio("Processing step", list(views), index=4, horizontal=True)
                draw_grid(views[selected_view], selected_view, show_values, cell_size,
                          rescale and not (stage is stages[0] and selected_view == "Input"))
                st.caption("CNN convolution uses cross-correlation: the kernel is not flipped.")
                st.dataframe(KERNELS[stage.layer.kernel], use_container_width=True)
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
