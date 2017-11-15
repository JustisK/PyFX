# PyFX
Inception-based feature extraction for image-based datasets. Written in Python3.

## Instructions
As a standard Python script, `PyFX` can be run from terminal or invoked using `system()` within C++ (or equivalent modes of shell interaction).

### Arguments / Invocation
Right now, only `extract_single` functions.  
Use `python pyfx.py path-to-file.extension`

### Usage notes
As of now, the script uses an 'unsafe' implementation, assuming correct input format for the target images and command line arguments. When using `PyFX`:
 * Do not put quotation marks around parameters.
 * Do not put '.' before the file extension parameter.
Failure to follow these guidelines will result in program failure and (potentially) corrupted output. Moreover, suitability checks aren't yet performed prior to initializing the model, so the program will spend a fair amount of time walking through the image directory before throwing any errors.

## Credit/Thanks
Thanks to @aleozlx (Alex Yang) for top-level model design (used in transfer learning exercise elsewhere).
