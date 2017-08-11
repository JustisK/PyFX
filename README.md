# PyFX
Inception-based feature extraction for image-based datasets. Written in Python3.

## Instructions
As a standard Python script, `PyFX` can be run from terminal or invoked using `system()` within C++ (or equivalent modes of shell interaction). 

### Arguments
`PyFX` takes seven (4+3) arguments: 
 * the extractor target - `multi` (multiple images) or `single` (single image)
 * a boolean value (bound to -s tag) for silent run (default is `True`)
 * an input path (e.g. directory containing images (e.g.for `extract_multi`) or 
 the address of a single image [for extract_single()]) - default is './images' 
 * an output path (e.g. directory and filename) - default is './output/features'
 * a **DOTLESS** file extension (e.g. `csv`, `txt`) for the stored features - default is `hdf5`
 * boolean `True` for compressed output, `False` for uncompressed - default is `False`
 * boolean `True` for flattened output, `False` for n-dimensional - default is `False`, 
 except for `csv` output (2d)
*Note - this is being streamlined in the near future.*
 
### Usage notes
As of now, the script uses an 'unsafe' implementation, assuming correct input format for the target images and command line arguments. When using `PyFX`:
 * Do not put quotation marks around parameters.
 * Do not put '.' before the file extension parameter. 
Failure to follow these guidelines will result in program failure and (potentially) corrupted output. Moreover, suitability checks aren't yet performed prior to initializing the model, so the program will spend a fair amount of time walking through the image directory before throwing any errors.

### Example invocation
`python pyfx.py -s True multi ./data/images ./output/fname csv'

* Reads multiple images 
* from {execution directory}/data/images
* then outputs to {execution directory}/output 
* in a file named `fname.csv`.
* By default: the file is *uncompressed*
* and (because of the CSV format) *flattened* to 1 dimension.

## License
The software is MIT-licensed.

## Credit/Thanks
Thanks to @aleozlx (Alex Yang) for top-level model design (used in transfer learning exercise elsewhere).
