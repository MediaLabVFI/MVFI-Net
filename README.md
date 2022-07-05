# MVFI-Net (Paper ID:)
MVFI-Net: Motion-aware Video Frame Interpolation Network 

## Requirements
* Linux
* python3.8
* pytorch1.7.0+cuda11.0
* cupy-cuda11.0
* pillow8.4.0

## Demo run
We only show the codes of MAC which is our key warping algorithm. Eq. (2) and Eq. (10) are reproduced.    
The rest of codes will be made publicly available soon.
* cd Demo
* python run.py  
Then f<sub>10</sub>, f<sub>12</sub> and I_bar will be generated and stored into './results'. Note that I_bar is not the final output of MVFI-Net.
