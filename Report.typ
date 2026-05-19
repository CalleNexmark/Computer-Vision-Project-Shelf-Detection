
// ---------- Document setup -----------------------------------------------------
#set document(title: "Product Recognition on Store Shelves", author: "Carl Nexmark")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

// Monospace style for code-like content (spec output excerpts).
#let speclisting(body) = block(
  fill: luma(245),
  stroke: 0.5pt + luma(180),
  inset: 8pt,
  radius: 3pt,
  width: 100%,
  text(font: ("DejaVu Sans Mono", "Menlo", "Courier New"), size: 9pt, body),
)

// Figure helper — short form for inline images.
//   #fig("figures/foo.png", [Caption text.])
//   #fig("figures/foo.png", [Caption text.], width: 60%)
#let fig(path, caption-body, width: 80%) = figure(
  image(path, width: width),
  caption: caption-body,
)


// ---------- Title page ---------------------------------------------------------
#align(center)[
  #v(4cm)
  #text(size: 22pt, weight: "bold")[Product Recognition on Store Shelves]
  #v(0.4em)
  #text(size: 13pt)[Computer Vision project · University of Bologna]
  #v(3cm)
  #text(size: 12pt, weight: "bold")[Carl Fredrik Cornelis Nexmark]
  #v(0.3em)
  #text(size: 11pt)[18 June 2026]
]
#pagebreak()



// ---------- Introduction -------------------------------------------------------
= Introduction

This project was carried out as part of a Computer Vision examination and investigates object detection techniques in supermarket environments. The main objective is to design a system capable of identifying products placed on store shelves through the use of computer vision methods.


Given an image of a supermarket shelf, the system is expected to recognize and localize the products present in the scene. 


The specific task addressed in this project is the detection of cereal boxes from different brands using a reference image for each product. For every detected product instance, the system must determine the number of occurrences, estimate the dimensions of the corresponding bounding box in pixels, and calculate the position of the object within the image using the center coordinates of the bounding box.


Object recognition in real-world environments requires methods that can handle variations in viewing conditions, including differences in scale, orientation, and lighting. Local invariant features address these challenges by enabling reliable identification of distinctive image characteristics despite such transformations. A complete recognition framework typically consists of detecting key features, generating descriptive representations, matching corresponding features between images, and estimating object position. Together, these steps form an effective pipeline for detecting and recognizing objects in images and video data.



// ---------- Task A -------------------------------------------------------------
= Task A — Multiple Product Detection

Task A works with the easy shelves `e1`–`e5`, where each scene contains at most one instance of each product. The product set is $\{0, 1, 11, 19, 24, 25, 26\}$, shown in @taska-products.

#figure(
  image("figures/product_taskA.png", width: 90%),
  caption: [The seven product reference images used in Task A (and reused in Task B).],
) <taska-products>

The pipeline therefore needs to do one thing well. Given a model image and a scene image, decide whether the product is present and, if so, return its bounding box. To solve this, SIFT keypoint matching followed by RANSAC homography estimation was used. This is a classical local-feature pipeline that fits the problem because the printed face of a cereal box is essentially planar.

== Feature extraction and matching

Keypoints and descriptors on both the model and the scene are extracted with SIFT. SIFT detects blob-like keypoints at extrema of the Difference-of-Gaussians scale-spacc. Each keypoint carries a position, a characteristic scale, and a dominant orientation. Around each keypoint, gradients are sampled from a normalised patch into a $128$-D histogram descriptor that is L2-normalised, clipped at $0.2$, and re-normalised. This gives invariance to affine intensity changes and robustness to non-linear ones, on top of the rotation invariance provided by the canonical orientation. SIFT was chosen because cereal-box covers are rich in texture and text, which is likely type of content that produces stable, repeatable keypoints, and the descriptor's invariances cover the kinds of pose and lighting variation seen across shelves.

Model keypoints are matched to scene keypoints with FLANN, an approximate nearest-neighbour library. It applies Best-Bin-First search over randomised k-d trees, which is what makes it tractable at the $128$-D SIFT descriptor size. For every model descriptor FLANN returns the two closest scene descriptors at distances $d_1 lt.eq d_2$. Lowe's ratio test then keeps a match only if $d_1 < 0.6 dot d_2$, f the second-closest match is comparably close, the first is ambiguous and likely wrong. A treshold value was found at $0.6$ from empirical tuning.

If fewer than $40$ matches survive the ratio test, the (model, scene) pair is skipped entirely, since RANSAC with very few matches is unlikely to find a good homography and more likely to produce false positives. This treshold is intentionally high, cause when combined with the strict ratio above, it makes the precision/recall trade-off skew well toward precision before RANSAC is even invoked.

== Geometric verification

After the ratio test some matches are still wrong. RANSAC is used to fit a planar homography to the surviving point correspondences. It repeatedly samples four match pairs at random, estimates the homography they imply, and counts inliers, matches whose reprojection error under the estimated homography is below $5$ px. After many iterations the homography with the largest inlier set is returned. A homography is the right model here, since each cereal-box face is essentially planar under a perspective camera, so the estimate both yields an accurate bounding box and serves as a geometric _verification_ of the matches. A coherent box produces a homography with many consistent supporters.

A detection is accepted only when at least $15$ matches are RANSAC inliers. This threshold is the main false-positive guard. When it is met, the four corners of the model image are warped through the estimated homography to obtain an axis-aligned bounding box.

== Colour check

To differentiate between objects with similar design but different colours, such as products `1` and `11`, a colour check was added based on the Euclidean distance between the BGR histograms of the model and the detected region in the scene. Other options such as the Bhattacharyya and Mahalanobis distances, mentioned in lecture notes, were considered, but Euclidean distance was chosen for its simplicity and effectiveness. The comparison is performed in the BGR colour space, the default in OpenCV.

The threshold was determined empirically by comparing the Euclidean distances of three model images (`0`, `1`, `11`) against the same scene region (@hist-threshold). The distance for the correct match (`1`) was significantly lower than for the incorrect ones, and a threshold of $50$ separates them cleanly while allowing some variability in BGR mean values across scenes. A worthy mention is product 0 in this test, which got the highest distance, even though it has much of the the same color profile of the scene, cause of the matching product. The reason for this is unknown, and test results were not replicated with several runs on product 0 alone.  

#figure(
  image("figures/Euclidian_treshold_test.png", width: 80%),
  caption: [Euclidean distance between the BGR histograms of the detected region in the scene and the model images for products `0`, `1`, and `11`.],
) <hist-threshold>

== Iterative masking sweep

 To process a whole shelf it is wrapped in an _iterative masking sweep_. At each iteration the full single-pair pipeline is run for every remaining candidate product, the product with the most RANSAC inliers is accepted, its bounding box is filled with black in a working copy of the scene, that product is removed from the candidate set, and SIFT is recomputed on the masked scene before the next iteration. The loop terminates when no remaining product yields an accepted detection.

This is known as a greedy detection strategy, and it is a simple way to allow multiple products to be detected in the same scene without one product's detections interfering with another's. It is locally optimal, but not globally, a confidently wrong winner can lock out a correct competitor. More principled alternatives exist, which may likely give a better global solution.



== Results

The figures below show the accepted detections on each of the five easy shelves after the iterative masking sweep. Green rectangles mark the warped model corners drawn back onto the original scene.

#fig("results/A_results/SceneE1.png", [Task A — accepted detections on shelf `e1`.])
#fig("results/A_results/SceneE2.png", [Task A — accepted detections on shelf `e2`.])
#fig("results/A_results/SceneE3.png", [Task A — accepted detections on shelf `e3`.])
#fig("results/A_results/SceneE4.png", [Task A — accepted detections on shelf `e4`.]) <coco>
#fig("results/A_results/SceneE5.png", [Task A — accepted detections on shelf `e5`.])


== Discussion

*Known limitations.*

The colour check is what seperates visually similar products that differ in colour palette, and the cleanest success is on the Krave variants. The BGR-mean distance between the wrong Krave model and the scene region comfortably exceeds the threshold of $50$, and the incorrect match is rejected unambiguously. The Nesquik pair — Cioccomilk (product `0`) and DUO (product `26`) is the counter-example. The two boxes share the Nesquik mascot and a large fraction of the printed graphics, so SIFT descriptors and the fitted homography support both at the same shelf location. The colour palettes are also closer than the threshold accounts for, even though Cioccomilk carries a pink accent that distinguishes it visually, the BGR mean over the full bounding box averages out into something only marginally different from DUO, is well within the $50$-unit tolerance. The colour check therefore does _not_ separate the two reliably. This is were the iterative masking sweep makes the problem to solve easier, as soon as one of the two Nesquik models is accepted at a shelf location, the region is blacked out, and the second model can no longer fire there.


A different failure mode is observed on the Coco Pops product, where the homography itself goes wrong rather than the colour, seen from scene `e4` in @coco. The exact reason is not clear, one plausibility is that the line with the price tag confuses the homography estimation, making it leak onto neighbouring boxes.

// ---------- Task B -------------------------------------------------------------
= Task B — Multi-instance Detection

Task B works on the medium shelves `m1`–`m5`, where each scene can contain multiple instances of the same product. The product set is the same as Task A, $\{0, 1, 11, 19, 24, 25, 26\}$. Task A's pipeline is fundamentally limited here, since RANSAC returns at most one homography per pair, so it cannot find more than one instance per product. Task A's SIFT, FLANN, and Lowe's ratio test are kept in Task B. The geometric verification step is replaced with a Generalised Hough Transform that votes for product locations in the scene.

== Voting with a star model

Feature extraction and matching follow Task A in shape but loosen the threshold slightly. SIFT keypoints and descriptors are extracted on the model and the scene, matched with FLANN at $k = 2$, and filtered with Lowe's ratio test. 

For each model keypoint at position $p_i$ a _joining vector_ is precomputed from the keypoint to the model image's barycentre:

$ v_i = b - p_i, $

where $b = (W slash 2, H slash 2)$ is the centre of the model image. The set of model features together with their joining vectors is what is known as a _star model_, and it replaces the R-Table used in the classical Generalised Hough Transform. At test time, every match between model keypoint $i$ and scene keypoint $j$ at position $p_j$ casts a vote for the model's barycentre location in the scene:

$ hat(b)_(i j) = p_j + frac(sigma_j, sigma_i) thin v_i, $

where $sigma_i$ and $sigma_j$ are the SIFT keypoint sizes. The ratio $sigma_j slash sigma_i$ absorbs the change in apparent scale between model and scene, so the same product is detected whether it appears at the front or the back of the shelf.

This variant was chosen over the classical R-Table because the front-end already produces rich, scale- and orientation-aware SIFT features. Storing per-feature offsets to the barycentre is both more expressive and a closer fit to what the front-end provides than rebuilding an edge-based R-Table on top of SIFT keypoints.

== Peak detection and bounding boxes

Votes are accumulated into a $2$-D histogram over the scene, binned at $10$ px per cell. Local maxima are extracted by comparing the accumulator against a max-filtered copy (`cv2.dilate`). A bin is a peak if its value equals the local max and exceeds $8$ votes. `cv2.dilate` is used for the peak search because it gives an exact local-max test in a single vectorised pass, which is more efficient than scanning the accumulator by hand and equivalent in outcome.

Within a suppression radius of $150$ px around each peak the matches that voted for it are gathered. The median of their scale ratios $sigma_j slash sigma_i$ is taken, and a bounding box of $W_("model") times "median scale"$ by $H_("model") times "median scale"$ is produced, centred on the peak. The median is used rather than the mean because it survives a handful of outlier voters cleanly, which an arithmetic mean does not.

#fig("figures/heatmap.png", [
  GHT accumulator for product `24` in scene `m1`. Two clear local maxima correspond to the two box instances.
], width: 70%)

== Results

The figures below show the accepted detections on each of the five medium shelves after voting, the colour check, and the iterative masking sweep. Green rectangles mark each detected instance's bounding box. The product label is drawn at the centre.

#fig("results/B_results/SceneM1.png", [Task B — accepted detections on shelf `m1`.])
#fig("results/B_results/SceneM2.png", [Task B — accepted detections on shelf `m2`.])
#fig("results/B_results/SceneM3.png", [Task B — accepted detections on shelf `m3`.])
#fig("results/B_results/SceneM4.png", [Task B — accepted detections on shelf `m4`.])
#fig("results/B_results/SceneM5.png", [Task B — accepted detections on shelf `m5`.])

== Discussion

The colour check from Task A is reused unchanged here — same Euclidean BGR-mean distance, same threshold of $50$. The intuition that a product's colour palette is intrinsic to the product, not to the scene, carries over. The check has more work to do in Task B because each (model, scene) pair can produce several candidate bounding boxes from the GHT peaks, but no per-scene retuning was needed. How effectively the colour check really separates the products in Task B is not clear or quantified, but the same arguments from Task A apply, and the visual results look good.

The same iterative-masking sweep as Task A wraps the per-pair pipeline, with two adjustments. First, the per-pair stage returns a _list_ of accepted instances rather than a single match, one bounding box per GHT peak that survives the colour check. Second, the arbitration score is `len(accepted) × len(good)` rather than RANSAC inlier count, since RANSAC is no longer in the loop. When a product wins an iteration, all of its accepted instances are committed to the detection list at once, and their bounding boxes are masked out in the working scene before the next iteration. The product of the two counts is used because either factor alone doesn't work. Tuning the PEAK_SUPPRESSION_RADIUS was neccesary to not overlap peaks from different instances of the same product, specificly `p0` and `p26`. The value of $150$ px was found empirically to give good results across all shelves.


// ---------- Conclusion ---------------------------------------------------------
= Conclusion

This project implemented a layered SIFT-based pipeline that covers both single-instance detection (Task A) and multi-instance detection (Task B), sharing a SIFT + FLANN + Lowe front-end and a Euclidean BGR-mean colour check. The two tasks diverge at the verification stage, where task A uses a single RANSAC homography, while Task B replaces it with a star-model GHT.

Detections are correct on all easy and medium scenes, including shelves with multiple copies of the same product. The biggest limitation and failure observed is the homography on product `24` in scene `e4`, which is likely caused by the price tag line confusing the geometric verification.

Natural extensions include a per-region HSV or LAB colour histogram in place of a single BGR mean, voting in a 4-D pose space $(t_x, t_y, theta, s)$ rather than only translation. More principled multi-instance detection strategies could be explored, such as non-max suppression on the GHT accumulator or a global inference step that considers all peaks and their supporters together rather than greedily accepting one at a time.

// ---------- References ---------------------------------------------------------
= References

#set enum(numbering: "[1]")

+ L. D. Stefano, "Local features part 1" University of Bologna, Computer Vision and Image Processing, 2025. Accessed: 2026-05-18.
+ L. D. Stefano, "Local features part 2" University of Bologna, Computer Vision and Image Processing, 2025. Accessed: 2026-05-18.
+ L. D. Stefano, "Instance Detection" University of Bologna, Computer Vision and Image Processing, 2025. Accessed: 2026-05-16.
+ O. Documentation, “OpenCV Documentation”, OpenCV Docs, 2026. Accessed: 2026-05-15 [Online]. Available: https://docs.opencv.org/4.x/

#set enum(numbering: "1.")
