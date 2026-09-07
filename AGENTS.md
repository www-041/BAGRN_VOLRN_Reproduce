\# Project Goal



Reproduce the BAGRN-VOLRN radiometric normalization method from the paper:

"Joint block adjustment and variational optimization for global and local radiometric normalization toward multiple remote sensing image mosaicking".



Use Python first, not C++.



Input:

\- Multiple geometrically aligned GeoTIFF remote sensing images.

\- Images may have overlap areas and radiometric differences.



Output:

\- Radiometrically normalized GeoTIFF images.

\- Optional mosaic preview.



Main modules:

1\. Read and write GeoTIFF images.

2\. Detect overlap areas from geospatial bounds.

3\. Compute mean and standard deviation in overlap areas.

4\. Implement BAGRN global radiometric normalization.

5\. Implement VOLRN local radiometric normalization.

6\. Implement evaluation metrics: ADM, ADSD, CD, GL, RDOA, Ave.

7\. Provide a command line script.



Coding requirements:

\- Use rasterio, numpy, scipy, shapely, tqdm.

\- Keep functions small and testable.

\- Do not invent formulas beyond the paper.

\- Add comments explaining each formula.

\- Add simple synthetic tests before using real remote sensing images.

