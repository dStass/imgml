# imgml

This package is designed to run a few different image inpainting algorithms including:

- NaiveLinearInpainter
- IntermediateLinearInpainter
- ExemplarInpainter

as detailed in the submitted report.

## Section 1: An overview

This package allows the user to load an image of choice, this image is then resized to 256 x 256 and
opens a graphical interface. This is hard-coded and require changing inside image_processing.py if the user
wish to increase this. It is not recommended as images larger than this will take much longer to inpaint, more on this later.

The user can use their mouse/cursor (highlighted in pink) and click-and-drag on areas of the image they want to remove,
the program will behave as other painting programs available online.

This will draw a pink circle the size of the cursor directly on the image. There are currently no ways to undo once
an area has been drawn on except by resetting the canvas, this can be achieved by pressing the R key.

Once an area has been highlighted, the user can then press the I key (upper-case i), to initiate the inpainting process.
Using the three inpainting methods, the inpainted attempt by each technique will be opened in a new window.

Some configuration settings can be modified using config.json in the root folder of the project.

To run the package, run this from the command line without any arguments:

```{python}
python app.py
```

## Section 2: Configuration Settings

Configuration settings are stored in a .json file called config.json at the root of the project.

A compiled list containing all fields in this file:

- IMG_PATH: path of the image to be loaded
- OUTPUT_FOLDER: where saved outputs are directed to
- OUTPUT_NAME: a name that saved files are based on
- PROGRAM_MODE
  - 0: The program will inpaint regions of the image that is highlighted by the pink cursor.
       RECOMMENDED: leave PROGRAM_MODE set to 0
  - 1: The program will inpaint over the pink highlighted region, it will then compare this to
       the underlying image. The motivation here is that the highlighted region can act as faults
       to the image and the algorithms will try to remove these faults to restore the image.
       The difference to mode 0 is that a comparison will be made between the inpainted image and the
       original image and a sum of residuals between both images returned for analysis purposes.
- layers: a list containing what additional layers are included, options are combinations of: 'canny' and 'entropy'.
  [RECOMMENDED]: leave as ['canny', 'entropy']
- inpaint: for each sub-fields, the user can toggle
  - show: after an image is inpainted, this toggles whether or not it be displayed
  - save: after an image is inpainted, this toggles whether or not it will be saved
  - test: used explicitly for analysis purposes in analysis.py [RECOMMENDED]: leave as false

  Subfields includes whether the user wants the mask displayed, or each subsequent methods of inpainting to be conducted and displayed.

  The user may only want to test each method one at a time.

  [IMPORTANT]: Leave the structure of this section as is, only modify true/false values.
- load_kmeans: as the image is loaded, a series of images will also be loaded as per the settings dictate. These other images are the
  image with reduced dimensionality in the number of colours. The fields are:
  - start: the number of colours the image is first constrained to satisfy
  - end: the minimum number of colours the image is constrained to satisfy
  - step: the difference/jump in number of bins images are constrained to

  Example: for start=20, end=4, step=8, the image will first be loaded normally, followed by a transformation of the same image constrained to 20 colours as determined by the kmeans clustering algorithm. This is then followed by an image with 20-8=12 colours, then 12-8=4 colours. The loop terminates once this becomes strictly lower than end.

- cursor: contains information regarding the cursor of the user.
- analysis: contains details for analysis.

## Section 3: Keyboard instructions

A compiled list containing the keys the user needs to know to navigate the package:

- [ 'L' ] : toggles the layers of an image without affecting the draw tool. The user can toggle between the image and canny/entropy layers of the same image.
- [ ',' / '<' ] : moves the canvas to the image located to the 'left' (based on loading order) of the current image if possible,
  this operation will clear the canvas. The user can use this to view different loaded kmeans-clustered images.
- [ '.' / '>' ] : moves the canvas to the iamge located to the 'right' of the current image if possible.
- [ '[' / '{' ] : decreases cursor size (if above min cursor size).
- [ ']' / '}' ] : increases cursor size (if below max cursor size).
- [ 'R' ] : resets the canvas and clears all drawn points.
- [ 'I' ] : starts the inpainting process, the application may seem like it HANGS or FREEZES. Do not worry, it just takes some time to run.
  Some inpainters take longer than others (such as the exemplar inpainter) and the larger the chosen region, the longer it takes.
- [ 'Q' ] : quits out of the application

## Section 4: Navigating the codebase

The driver code for the graphical interface can be found in app.py.

All relevant code for image related operations can be found in inside the sub-package processing.

### Edge-detection

Related code can be found in edge_processing.py.

There are two main classes:

1. EntropyEdgeDetection: utilises kmeans clustered images and uses the Shannon entropy theory to look at variations local regions. Regions with sharp edges are highlighted compared to other dull areas. This is accomplished by calling **generate_binary_edges_heatmap**. Some further care is taken to differentiate dull subregions by using a ~65% confidence interval (one standard deviation from the mean) and do not count them as points to use as edges (white points). A binary image is then produced using qualifying points (any points higher than -1 standard deviation from the mean) as white and the rest as black points.

2. CannyEdgeDetection: employs cv2's Canny edge detection algorithm to return a Canny edge image.

### Inpainting

Related code can be found in inpaint_processing.py.

There are three main classes:

1. NaiveLinearInpainter: 
2. IntermediateLinearInpainter
3. ExemplarInpainter