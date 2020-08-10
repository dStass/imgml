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

There are three main classes, where each takes the image and a mask as a guide to inpainting.

1. NaiveLinearInpainter: Implementation of the first method outlined in the paper.
   This method finds all non-masked points directly neighbouring masked points and uses those points as training data points for a linear regression of
   two variables (row and column) with the response being the colour (between 0 and 255). This is split across all three channels R G B separately as
   we assume these models occur independently of each other. No formal statistical tests have been tested to verify this assumption.
   This will generate the best fitting gradient plane by least squares and fill the mask.

2. IntermediateLinearInpainter: A similar method, however, instead of utilising **all** surrounding neighbours to the masked region, we only consider those that exists with at least one same coordinate. An example is given a point to inpaint X at (row_0, col_0), we only consider the four points with two directly to the left and right of X with a fixed row at row_0 and similarly the other two directly above and below with fixed column at col_0. This method on average, yields better results when compared to method 1.

3. ExemplarInpainter: A different approach, this works by patching sub-regions (7x7 pixel blocks) one by one by prioritising patches in some way and then comparing the patch with other patches, finding the most similar patch and copy the pixel values across for the pixels requiring inpainting. The method chosen to prioritise regions in this project is simply based on how much information in the form of edges exists in the region. This is adapted from the literature where linear structures are prioritised as this is heavily impactful to the solution. Within the literature however, directional gradients and orthogonal vectors at pixels are considered. This proved to be too difficult to implement (see denoise.py for attempts at implimenting image gradient processes).
Once the most prioritised patch is found, we compare it to other candidate patches in the image and pick the one that minimises the square sum of residuals (of each channel in each pixel) of the entire patch. This method is quite computationally expensive and takes a lot longer compared to methods 1 and 2, especially for large images or large patch sizes.

## Section 5: Analysis

The file analysis.py details a method for randomly producing masks and comparing an inpainted image to the original using the squared sum of residuals described above. This will then automatically write to a csv file with the information used to plot the charts in the report.