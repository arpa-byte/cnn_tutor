# CNN Teaching Lab

An interactive dashboard for learning the mechanics of convolutional neural networks. Use it to follow a filter over an image, explore how a stack of operations changes a feature map, and practise calculating feature-map sizes by hand.

## Start the lab

You need Python 3.10 or newer. Complete this setup once from the repository folder.

```bash
./install
```

Then start the dashboard with:

```bash
./start
```

The launcher starts the dashboard in the background and opens it at [http://localhost:8501](http://localhost:8501). Closing the dashboard tab stops its local server shortly afterwards, so it does not remain running in the background.

On macOS, you can also double-click **CNN Tutor.app** in Finder. Its CNN icon runs the same setup and launcher.

## Platform support

The included launchers are designed for macOS. Linux and Windows need platform-specific launch scripts before students can use the one-step launch workflow.

If the terminal reports `permission denied`, run this once and try again:

```bash
chmod +x install start
```

## What to explore

### Image processing

Choose an input pattern and inspect common image operations.

- Move through convolution and pooling one output pixel at a time.
- Edit a 3 × 3 filter and observe the resulting feature map.
- Compare VALID and SAME padding, and change stride where available.
- Use **Show only output** when you want to compare just the input and completed result.

The inputs are grayscale pixels from 0 to 255. Calculated feature maps use floating-point values, so filter responses can be negative or larger than 255. The contrast-rescale control changes only the display.

### CNN network builder

Build a sequential feature extractor from fixed filters. The restored example has ten processing blocks. Each block performs preprocessing, convolution, activation, and optional pooling.

- Select a block to edit its filter, padding, stride, activation, and pooling settings.
- Add, duplicate, remove, and reorder blocks.
- Use the player to see an operation produce its output, then hand that output to the next operation.
- Start from an empty network to construct your own sequence.

This tab demonstrates feature extraction. It does not train a model or make class predictions.

### CNN size practice

Generate a random CNN and calculate the output feature-map size after every layer.

1. Start with the displayed input size `H × W × C`.
2. For each layer, enter its output height, width, and channel count.
3. Select **Check my calculations** to find the first incorrect layer.
4. Open the worked answers only after you want to review the solution.

The flow chart shows the order of layers and their settings without revealing the sizes to calculate. For spatial dimensions, SAME padding gives `ceil(input / stride)`. VALID padding gives `floor((input - kernel) / stride) + 1`. A convolution’s output channels equal its number of filters, while max pooling keeps the channel count unchanged.

## Optional checks

Run the included project checks with:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python architecture_practice_checks.py
```

The playback-control checks require Node.js.
